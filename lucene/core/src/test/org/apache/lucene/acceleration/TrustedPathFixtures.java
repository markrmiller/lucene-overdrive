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

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.SuppressForbidden;

/** Real trusted installation fixtures do not inherit writable Gradle build directories. */
final class TrustedPathFixtures {
  private TrustedPathFixtures() {}

  /**
   * Trust checks read ownership up to "/", process credentials and process-wide properties, which
   * the test sandbox denies. Acceleration needs JDK 25, where no security manager can be installed.
   */
  @SuppressForbidden(reason = "security manager")
  @SuppressWarnings("removal")
  static void assumeHostAccess() {
    LuceneTestCase.assumeTrue(
        "acceleration trust checks need host access outside the test sandbox",
        System.getSecurityManager() == null);
  }

  static Path directory(String prefix) throws IOException {
    assumeHostAccess();
    Path path =
        Files.createTempDirectory(
            Path.of("/tmp"),
            "lucene-trusted-" + prefix + "-",
            PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rwx------")));
    LuceneTestCase.closeAfterSuite((Closeable) () -> IOUtils.rm(path));
    return path;
  }

  static Path file(String prefix) throws IOException {
    return directory(prefix).resolve("policy.tsv");
  }
}
