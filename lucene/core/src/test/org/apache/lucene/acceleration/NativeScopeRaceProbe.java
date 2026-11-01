/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.acceleration;

import java.lang.reflect.InvocationTargetException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;

/** Isolated real-native large downcall race; never runs in a server JVM. */
public final class NativeScopeRaceProbe {
  private NativeScopeRaceProbe() {}

  public static void main(String[] arguments) throws Exception {
    Class<?> arenaApi = Class.forName("java.lang.foreign.Arena");
    Class<?> segmentApi = Class.forName("java.lang.foreign.MemorySegment");
    Class<?> scopeApi = Class.forName("java.lang.foreign.MemorySegment$Scope");
    NativeBridge bridge =
        (NativeBridge)
            Class.forName("org.apache.lucene.internal.acceleration.FfmBridge")
                .getConstructor()
                .newInstance();
    var ledger = new MemoryLedger(new MemoryLedger.Limits(0, 256L << 20, 0), 128L << 20);
    var kernel =
        bridge
            .prepare(
                PackLoader.verify(Path.of(arguments[0])).library(),
                new AccelerationProvider.Descriptor(
                    "scope-race", "1", 1, Set.of(Capability.DOCSET_INTERSECT)),
                ledger)
            .bind(Capability.DOCSET_INTERSECT);
    int words = 4 * 1024 * 1024;
    long inputBytes = (long) words * Long.BYTES;
    if (NativeBridge.criticalCallAllowed(Capability.DOCSET_INTERSECT, inputBytes * 3))
      throw new AssertionError("fixture must use ordinary downcall");
    int closeRejections = 0, successes = 0, closedBeforeEntry = 0;
    for (int attempt = 0; attempt < 12 && closeRejections == 0; attempt++) {
      Object arena = arenaApi.getMethod("ofShared").invoke(null);
      Object left =
          arenaApi.getMethod("allocate", long.class, long.class).invoke(arena, inputBytes, 8L);
      Object right =
          arenaApi.getMethod("allocate", long.class, long.class).invoke(arena, inputBytes, 8L);
      Object inputScope = segmentApi.getMethod("scope").invoke(left);
      segmentApi.getMethod("fill", byte.class).invoke(left, (byte) -1);
      segmentApi.getMethod("fill", byte.class).invoke(right, (byte) 0x5a);
      try (var owner =
          new RequestMemoryScope(
              ledger,
              new MemoryLedger.Owner("native-race", "isolated", 0, "native-race"),
              bridge)) {
        ByteBuffer output = owner.allocate(inputBytes).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer parameters = owner.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
        parameters.putLong(0, words).putLong(16, (long) words * 64);
        var batch =
            new AccelerationProvider.Batch(
                Capability.DOCSET_INTERSECT,
                List.of(
                    new AccelerationProvider.Input(
                        (ByteBuffer) segmentApi.getMethod("asByteBuffer").invoke(left),
                        words,
                        AccelerationProvider.Encoding.I64),
                    new AccelerationProvider.Input(
                        (ByteBuffer) segmentApi.getMethod("asByteBuffer").invoke(right),
                        words,
                        AccelerationProvider.Encoding.I64)),
                new AccelerationProvider.Output(output, words),
                parameters,
                owner.allocate(0),
                owner.allocate(256));
        // Warm both the FFM entry and cached views, so link/setup work cannot fake the overlap.
        for (int warm = 0; warm < 3; warm++) requireSuccess(kernel.execute(batch), words, output);
        CountDownLatch entering = new CountDownLatch(1);
        AtomicBoolean rejected = new AtomicBoolean();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        AtomicReference<AccelerationProvider.Result> completed = new AtomicReference<>();
        Thread worker =
            new Thread(
                () -> {
                  entering.countDown();
                  try {
                    completed.set(kernel.execute(batch));
                  } catch (Throwable error) {
                    failure.set(error);
                  }
                },
                "ordinary-native-call");
        worker.start();
        if (!entering.await(10, TimeUnit.SECONDS))
          throw new AssertionError("native worker did not enter");
        // Closing too early or too late is a valid case, but cannot satisfy this regression.
        LockSupport.parkNanos(100_000);
        try {
          arenaApi.getMethod("close").invoke(arena);
        } catch (InvocationTargetException close) {
          if (!(close.getCause() instanceof IllegalStateException)) throw close;
          if (!(boolean) scopeApi.getMethod("isAlive").invoke(inputScope))
            throw new AssertionError("rejected close freed input scope", close);
          rejected.set(true);
          closeRejections++;
          System.out.println(
              "CLOSE_REJECTED_DURING_ORDINARY_DOWNCALL " + close.getCause().getMessage());
        }
        worker.join(30000);
        if (worker.isAlive()) throw new AssertionError("ordinary native call exceeded deadline");
        Throwable error = failure.get();
        if (rejected.get()) {
          if (error != null)
            throw new AssertionError("held native input failed after rejected close", error);
          requireSuccess(completed.get(), words, output);
          successes++;
          arenaApi.getMethod("close").invoke(arena);
          if ((boolean) scopeApi.getMethod("isAlive").invoke(inputScope))
            throw new AssertionError("retirement retry left scope alive");
          System.out.println("RETIREMENT_RETRY_SUCCEEDED");
          try {
            kernel.execute(batch);
            throw new AssertionError("closed native input executed");
          } catch (
              @SuppressWarnings("unused")
              IllegalStateException expected) {
            closedBeforeEntry++;
          }
        } else if (error == null) {
          requireSuccess(completed.get(), words, output);
          successes++;
        } else if (error instanceof IllegalStateException) {
          closedBeforeEntry++;
        } else throw new AssertionError("unexpected native race failure", error);
      } finally {
        if ((boolean) scopeApi.getMethod("isAlive").invoke(inputScope))
          arenaApi.getMethod("close").invoke(arena);
      }
    }
    if (closeRejections == 0)
      throw new AssertionError("no actual in-flight scope acquisition witnessed; outcome unknown");
    if (ledger.snapshot().charged().values().stream().anyMatch(bytes -> bytes != 0))
      throw new AssertionError("owned native output remained charged");
    System.out.println(
        "NATIVE_SCOPE_RACE successes="
            + successes
            + " closeRejections="
            + closeRejections
            + " closedRejections="
            + closedBeforeEntry);
    System.out.println("DONE_NATIVE_SCOPE_RACE rc=0");
  }

  private static void requireSuccess(
      AccelerationProvider.Result result, int words, ByteBuffer output) {
    if (result == null
        || result.status() != AccelerationProvider.Status.OK
        || result.outputElements() != words
        || result.detail() != 0)
      throw new AssertionError("incomplete native intersection: " + result);
    for (int index : new int[] {0, words / 2, words - 1})
      if (output.getLong(index * Long.BYTES) != 0x5a5a5a5a5a5a5a5aL)
        throw new AssertionError("native intersection output mismatch");
  }
}
