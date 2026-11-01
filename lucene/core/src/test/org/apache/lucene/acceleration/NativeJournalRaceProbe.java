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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;

/** Multiple isolated JVMs repeatedly create/recover/close journals in one private directory. */
public final class NativeJournalRaceProbe {
  private NativeJournalRaceProbe() {}

  public static void main(String[] arguments) throws Exception {
    Path directory = Path.of(arguments[0]);
    int participant = Integer.parseInt(arguments[1]), participants = Integer.parseInt(arguments[2]);
    Files.writeString(directory.resolve("ready-" + participant), "ready");
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
    for (; ; ) {
      int ready = 0;
      for (int i = 0; i < participants; i++)
        if (Files.exists(directory.resolve("ready-" + i))) ready++;
      if (ready == participants) break;
      if (System.nanoTime() > deadline)
        throw new AssertionError("journal race peers did not start");
      LockSupport.parkNanos(100_000);
    }
    String key = NativeCallJournal.key("journal-process-" + participant);
    for (int loop = 0; loop < 100; loop++) {
      try (var journal = new NativeCallJournal(directory)) {
        int slot = journal.begin(key, Capability.OSQ_GATHER, 32);
        NativeCallJournal.recover(directory);
        journal.end(slot);
        if (journal.quarantined(key))
          throw new AssertionError("live process was incorrectly quarantined");
      }
    }
    System.out.println("DONE_JOURNAL_PROCESS participant=" + participant + " cycles=100 rc=0");
  }
}
