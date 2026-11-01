/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.acceleration;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;

/**
 * Reusable native storage for one thread's synchronous kernel calls, owned by the runtime.
 *
 * <p>Per-call scopes pay ledger reservation, allocation and release on every batch. A frame is
 * admitted once per thread, so a call only fills its buffers. Obtain it with {@link
 * AccelerationRuntime#acquireFrame()} and close it when the call has published its output. Closing
 * releases the frame for the thread's next call and does not free it. A frame is not reentrant: a
 * nested acquire on the same thread returns null and the caller uses a per-call scope.
 *
 * <p>The runtime frees a frame at shutdown, or when a new frame is created after its thread has
 * exited. Its storage uses shared memory so that either can happen from another thread; that close
 * costs one VM handshake per frame lifetime rather than one per call.
 *
 * @lucene.experimental
 */
public final class CallFrame implements AutoCloseable {
  /** Bytes available to {@link #data()}: inputs and outputs for one call. */
  public static final int DATA_BYTES = 16 << 10;

  /** Bytes available to {@link #scratch()}. */
  public static final int SCRATCH_BYTES = 4 << 10;

  /** Bytes available to {@link #metadata()}. */
  public static final int METADATA_BYTES = 256;

  private static final int IDLE = 0, BUSY = 1, RETIRED = 2;

  private final AccelerationRuntime runtime;
  private final RequestMemoryScope scope;
  private final Thread thread;
  private final ByteBuffer data, parameters, scratch, metadata;
  private final AtomicInteger state = new AtomicInteger(IDLE);

  CallFrame(AccelerationRuntime runtime, RequestMemoryScope scope, Thread thread) {
    this.runtime = Objects.requireNonNull(runtime);
    this.scope = Objects.requireNonNull(scope);
    this.thread = Objects.requireNonNull(thread);
    data = scope.allocate(DATA_BYTES);
    parameters = scope.allocate(64);
    scratch = scope.allocate(SCRATCH_BYTES);
    metadata = scope.allocate(METADATA_BYTES);
  }

  /** The thread this frame belongs to. */
  public Thread thread() {
    return thread;
  }

  /** The owning scope, for calls that take an owner, such as shadow comparison. */
  public RequestMemoryScope scope() {
    return scope;
  }

  /** {@code bytes} of input/output storage starting at {@code offset}, little-endian. */
  public ByteBuffer data(int offset, int bytes) {
    Objects.checkFromIndexSize(offset, bytes, DATA_BYTES);
    return data.duplicate().position(offset).limit(offset + bytes).slice().order(ORDER);
  }

  /** The whole data region, little-endian. */
  public ByteBuffer data() {
    return data(0, DATA_BYTES);
  }

  /** The 64-byte parameter block, little-endian; callers write every field they use. */
  public ByteBuffer parameters() {
    return parameters.duplicate().order(ORDER);
  }

  /** {@code bytes} of kernel scratch. */
  public ByteBuffer scratch(int bytes) {
    Objects.checkFromIndexSize(0, bytes, SCRATCH_BYTES);
    return scratch.duplicate().limit(bytes).slice().order(ORDER);
  }

  /** The whole scratch region. */
  public ByteBuffer scratch() {
    return scratch(SCRATCH_BYTES);
  }

  /** {@code bytes} of kernel metadata. */
  public ByteBuffer metadata(int bytes) {
    Objects.checkFromIndexSize(0, bytes, METADATA_BYTES);
    return metadata.duplicate().limit(bytes).slice().order(ORDER);
  }

  /** The whole metadata region. */
  public ByteBuffer metadata() {
    return metadata(METADATA_BYTES);
  }

  /** Releases the frame for this thread's next call; the storage stays allocated. */
  @Override
  public void close() {
    if (Thread.currentThread() != thread)
      throw new IllegalStateException("call frame belongs to another thread");
    if (state.compareAndSet(BUSY, IDLE) && runtime.isStopping()) retire();
  }

  boolean acquire() {
    return state.compareAndSet(IDLE, BUSY);
  }

  boolean isRetired() {
    return state.get() == RETIRED;
  }

  /** Frees an idle frame; a busy frame is freed by its owner's close. */
  boolean retire() {
    if (!state.compareAndSet(IDLE, RETIRED)) return false;
    runtime.forgetFrame(this);
    scope.close();
    return true;
  }

  private static final ByteOrder ORDER = ByteOrder.LITTLE_ENDIAN;
}
