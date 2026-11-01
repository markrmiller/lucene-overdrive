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
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.SuppressForbidden;

public class TestNativeScopeRace extends LuceneTestCase {
  @SuppressForbidden(
      reason = "Subprocess isolation and ProcessBuilder directory/redirection require File")
  public void testLargeOrdinaryDowncallPinsScopeUntilReturn() throws Exception {
    Path pack = TestAccelerationRuntime.nativePack();
    Path directory = Path.of(createTempDir("native-scope-race").toString());
    Path output = directory.resolve("probe.log");
    String core =
        Path.of(
                AccelerationRuntime.class
                    .getProtectionDomain()
                    .getCodeSource()
                    .getLocation()
                    .toURI())
            .toString();
    String ffm =
        Path.of(
                Class.forName("org.apache.lucene.internal.acceleration.FfmBridge")
                    .getProtectionDomain()
                    .getCodeSource()
                    .getLocation()
                    .toURI())
            .toString();
    String tests =
        Path.of(
                NativeScopeRaceProbe.class
                    .getProtectionDomain()
                    .getCodeSource()
                    .getLocation()
                    .toURI())
            .toString();
    String classpath = String.join(System.getProperty("path.separator"), core, ffm, tests);
    var process =
        new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--enable-native-access=ALL-UNNAMED",
                "-XX:ErrorFile=" + directory.resolve("hs_err_pid%p.log"),
                "-cp",
                classpath,
                NativeScopeRaceProbe.class.getName(),
                pack.toString())
            .directory(directory.toFile())
            .redirectErrorStream(true)
            .redirectOutput(output.toFile())
            .start();
    if (!process.waitFor(90, TimeUnit.SECONDS)) {
      process.destroyForcibly();
      fail("native scope race subprocess timed out; outcome unknown");
    }
    String log = Files.readString(output);
    assertEquals(log, 0, process.exitValue());
    assertTrue(log, log.contains("CLOSE_REJECTED_DURING_ORDINARY_DOWNCALL"));
    assertTrue(log, log.contains("RETIREMENT_RETRY_SUCCEEDED"));
    assertTrue(log, log.contains("DONE_NATIVE_SCOPE_RACE rc=0"));
    try (var files = Files.list(directory)) {
      assertFalse(
          "native race emitted fatal JVM evidence",
          files.anyMatch(path -> path.getFileName().toString().startsWith("hs_err_pid")));
    }
    System.out.println(log);
  }
}
