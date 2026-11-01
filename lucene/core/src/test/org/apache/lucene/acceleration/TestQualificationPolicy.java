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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.List;
import java.util.Set;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestQualificationPolicy extends LuceneTestCase {
  private String row() {
    return AccelerationRuntime.BRIDGE_BUILD
        + "\t"
        + "a".repeat(64)
        + "\tx86-avx2-os-xsave-ymm\t"
        + Capability.OSQ_GATHER.id
        + "\t384\t8\t64\t1\t6\tbuffer\t65536\tboundary-qualification";
  }

  public void testDigestIsaAndExactShape() throws Exception {
    var policy = QualificationPolicy.parse(row());
    assertTrue(
        policy.forKernel("b".repeat(64), "x86-avx2-os-xsave-ymm", Capability.OSQ_GATHER).isEmpty());
    assertTrue(
        policy.forKernel("a".repeat(64), "baseline-scalar", Capability.OSQ_GATHER).isEmpty());
    var profile =
        policy.forKernel("a".repeat(64), "x86-avx2-os-xsave-ymm", Capability.OSQ_GATHER).getFirst();
    assertTrue(profile.shape(new AccelerationProvider.Shape(32, 384, 1000), false));
    assertFalse(profile.shape(new AccelerationProvider.Shape(32, 128, 1000), false));
    assertFalse(profile.shape(new AccelerationProvider.Shape(7, 384, 1000), false));
    assertFalse(profile.shape(new AccelerationProvider.Shape(65, 384, 1000), false));
    assertFalse(profile.shape(new AccelerationProvider.Shape(32, 384, 65537), false));
    var parameters =
        ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN).putLong(24, 1).putLong(32, 6);
    var batch =
        new AccelerationProvider.Batch(
            Capability.OSQ_GATHER,
            List.of(),
            new AccelerationProvider.Output(ByteBuffer.allocate(4), 1),
            parameters,
            ByteBuffer.allocate(0),
            ByteBuffer.allocate(256));
    assertTrue(profile.batch(batch));
    parameters.putLong(24, 0);
    assertFalse(profile.batch(batch));
  }

  public void testOptionalConsumerContextKeepsLegacyRowsGeneric() throws Exception {
    var legacy = QualificationPolicy.parse(row()).profiles.getFirst();
    assertTrue(legacy.context("generic"));
    assertFalse(legacy.context("solr.point.uncached.full-live.v1"));
    String rangeRow = row().replace(Capability.OSQ_GATHER.id, Capability.DOCVALUES_RANGE_FILTER.id);
    var tagged =
        QualificationPolicy.parse(rangeRow + "\tsolr.point.uncached.full-live.v1")
            .profiles
            .getFirst();
    assertTrue(tagged.context("solr.point.uncached.full-live.v1"));
    assertFalse(tagged.context("generic"));
    assertFalse(tagged.context("solr.frange.full-live.v1"));
    assertTrue(
        QualificationPolicy.parse(rangeRow + "\tsolr.frange.full-live.v1")
            .profiles
            .getFirst()
            .context("solr.frange.full-live.v1"));
    assertTrue(
        QualificationPolicy.parse(rangeRow + "\tgeneric").profiles.getFirst().context("generic"));
  }

  public void testMalformedConsumerContextsFailClosed() throws Exception {
    String rangeRow = row().replace(Capability.OSQ_GATHER.id, Capability.DOCVALUES_RANGE_FILTER.id);
    for (String invalid :
        List.of(
            "",
            "a".repeat(65),
            "point full live",
            "Point",
            "point/x",
            "point\textra",
            "unknown-context")) {
      expectThrows(
          java.io.IOException.class, () -> QualificationPolicy.parse(rangeRow + "\t" + invalid));
    }
  }

  public void testExplicitContextCannotEnrollUnscopedCapabilities() throws Exception {
    for (Capability capability : List.of(Capability.OSQ_GATHER, Capability.DOCSET_CARDINALITY)) {
      String unsupported = row().replace(Capability.OSQ_GATHER.id, capability.id);
      for (String context : List.of("generic", "solr.point.uncached.full-live.v1")) {
        expectThrows(
            java.io.IOException.class,
            () -> QualificationPolicy.parse(unsupported + "\t" + context));
      }
      assertTrue(
          "legacy 12-column rows retain generic compatibility",
          QualificationPolicy.parse(unsupported).profiles.getFirst().context("generic"));
    }
  }

  public void testByteContextsAreBoundedMappedAndCapabilitySpecific() throws Exception {
    String byteRow =
        row()
            .replace(Capability.OSQ_GATHER.id, Capability.VECTOR_I8.id)
            .replace("\tbuffer\t", "\tmapped\t");
    for (String context :
        List.of(
            OwnedByteBatchScorer.EXACT_OWNED_CONTEXT,
            OwnedByteBatchScorer.EXACT_STANDALONE_CONTEXT)) {
      var profile = QualificationPolicy.parse(byteRow + "\t" + context).profiles.getFirst();
      assertTrue(profile.context(context));
      assertFalse(profile.context("generic"));
      expectThrows(
          java.io.IOException.class,
          () ->
              QualificationPolicy.parse(
                  byteRow.replace("\tmapped\t", "\tnative\t") + "\t" + context));
      for (Capability other :
          List.of(
              Capability.OSQ_GATHER,
              Capability.DOCVALUES_AGGREGATE,
              Capability.DOCVALUES_RANGE_FILTER))
        expectThrows(
            java.io.IOException.class,
            () ->
                QualificationPolicy.parse(
                    byteRow.replace(Capability.VECTOR_I8.id, other.id) + "\t" + context));
    }
    assertTrue(QualificationPolicy.parse(byteRow).profiles.getFirst().context("generic"));
    assertTrue(
        QualificationPolicy.parse(byteRow + "\tgeneric").profiles.getFirst().context("generic"));
    for (String other : List.of("lucene.knn-byte.exact.v1", "solr.frange.full-live.v1"))
      expectThrows(
          java.io.IOException.class, () -> QualificationPolicy.parse(byteRow + "\t" + other));
  }

  public void testPackedStandaloneContextIsProducerVersioned() throws Exception {
    String byteRow =
        row()
            .replace(Capability.OSQ_GATHER.id, Capability.VECTOR_I8.id)
            .replace("\tbuffer\t", "\tmapped\t");
    String packed = "lucene.knn-byte.exact.standalone.packed.v2";
    String legacy = "lucene.knn-byte.exact.standalone.v1";
    QualificationPolicy policy;
    try {
      policy = QualificationPolicy.parse(byteRow + "\t" + packed);
    } catch (java.io.IOException rejected) {
      throw new AssertionError(
          "packed producer v2 context must be recognized by its matching core", rejected);
    }
    var profile = policy.profiles.getFirst();
    assertTrue(profile.context(packed));
    assertFalse(profile.context(legacy));
    assertFalse(profile.context(OwnedByteBatchScorer.EXACT_OWNED_CONTEXT));
    assertFalse(profile.context("generic"));
    assertTrue(
        QualificationPolicy.parse(byteRow + "\t" + legacy).profiles.getFirst().context(legacy));
    for (String layout : List.of("native", "buffer")) {
      expectThrows(
          java.io.IOException.class,
          () ->
              QualificationPolicy.parse(
                  byteRow.replace("\tmapped\t", "\t" + layout + "\t") + "\t" + packed));
    }
    for (Capability other :
        List.of(
            Capability.OSQ_GATHER,
            Capability.DOCVALUES_AGGREGATE,
            Capability.DOCVALUES_RANGE_FILTER)) {
      expectThrows(
          java.io.IOException.class,
          () ->
              QualificationPolicy.parse(
                  byteRow.replace(Capability.VECTOR_I8.id, other.id) + "\t" + packed));
    }
  }

  public void testAggregateContextIsBoundedMappedAndCapabilitySpecific() throws Exception {
    String aggregateRow =
        row()
            .replace(Capability.OSQ_GATHER.id, Capability.DOCVALUES_AGGREGATE.id)
            .replace("\tbuffer\t", "\tmapped\t");
    String context = "solr.json-range.full-live.v1";
    var profile = QualificationPolicy.parse(aggregateRow + "\t" + context).profiles.getFirst();
    assertTrue(profile.context(context));
    assertFalse(profile.context("generic"));
    for (String layout : List.of("native", "buffer"))
      expectThrows(
          java.io.IOException.class,
          () ->
              QualificationPolicy.parse(
                  aggregateRow.replace("\tmapped\t", "\t" + layout + "\t") + "\t" + context));
    for (Capability other :
        List.of(Capability.VECTOR_I8, Capability.OSQ_GATHER, Capability.DOCVALUES_RANGE_FILTER))
      expectThrows(
          java.io.IOException.class,
          () ->
              QualificationPolicy.parse(
                  aggregateRow.replace(Capability.DOCVALUES_AGGREGATE.id, other.id)
                      + "\t"
                      + context));
    for (String other : List.of("solr.json-range.unknown.v1", "solr.frange.full-live.v1"))
      expectThrows(
          java.io.IOException.class, () -> QualificationPolicy.parse(aggregateRow + "\t" + other));
    for (String layout : List.of("mapped", "native", "buffer")) {
      String legacy = aggregateRow.replace("\tmapped\t", "\t" + layout + "\t");
      assertTrue(QualificationPolicy.parse(legacy).profiles.getFirst().context("generic"));
      assertTrue(
          QualificationPolicy.parse(legacy + "\tgeneric").profiles.getFirst().context("generic"));
    }
  }

  public void testExactMaterializationPreflightMatchesOneWholeProfileWithoutSideEffects()
      throws Exception {
    Path pack = TestAccelerationRuntime.nativePack();
    Path policy = TrustedPathFixtures.file("materialization-policy");
    String prefix =
        AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + PackLoader.verify(pack).digest()
            + "\tx86-avx2-os-xsave-ymm\t"
            + Capability.DOCVALUES_RANGE_FILTER.id;
    Files.writeString(
        policy,
        prefix
            + "\t0\t4\t4\t-10\t10\tnative\t368\tunit-preflight-point\tsolr.point.uncached.full-live.v1\n"
            + prefix
            + "\t0\t8\t8\t-20\t20\tnative\t408\tunit-preflight-frange\tsolr.frange.full-live.v1\n");
    Files.setPosixFilePermissions(policy, PosixFilePermissions.fromString("rw-------"));
    String key = "lucene.acceleration.native.qualificationPath";
    String previous = System.getProperty(key);
    String stateKey = "lucene.acceleration.native.statePath",
        previousState = System.getProperty(stateKey);
    System.setProperty(key, policy.toString());
    System.setProperty(stateKey, TrustedPathFixtures.directory("materialization-state").toString());
    try (var runtime =
        new AccelerationRuntime(
            TestAccelerationRuntime.config(
                AccelerationConfig.Mode.AUTO, pack, Set.of(), Set.of(), 1))) {
      Capability capability = Capability.DOCVALUES_RANGE_FILTER;
      String point = "solr.point.uncached.full-live.v1", frange = "solr.frange.full-live.v1";
      var shape =
          new AccelerationProvider.Shape(
              4,
              0,
              368,
              Set.of(AccelerationProvider.Encoding.I64),
              AccelerationProvider.LayoutKind.NATIVE_BUFFER,
              8,
              true);
      var before = runtime.inventory();
      assertTrue(
          before.rejections().toString(), runtime.supports(capability, shape, -10, 10, point));
      assertFalse(runtime.supports(capability, shape, -10, 10, frange));
      assertFalse(
          "matching shape and bounds from different rows cannot be combined",
          runtime.supports(capability, shape, -20, 20, frange));
      assertFalse(runtime.supports(capability, shape, -11, 10, point));
      assertFalse(runtime.supports(capability, shape, -10, 11, point));
      assertFalse(runtime.supports(capability, shape, -10, 10, "generic"));
      assertFalse(runtime.supports(capability, shape, -10, 10, "bad context"));
      assertFalse(runtime.supports(Capability.DOCVALUES_AGGREGATE, shape, -10, 10, point));
      for (var rejected :
          List.of(
              new AccelerationProvider.Shape(
                  3,
                  0,
                  368,
                  Set.of(AccelerationProvider.Encoding.I64),
                  AccelerationProvider.LayoutKind.NATIVE_BUFFER,
                  8,
                  true),
              new AccelerationProvider.Shape(
                  5,
                  0,
                  368,
                  Set.of(AccelerationProvider.Encoding.I64),
                  AccelerationProvider.LayoutKind.NATIVE_BUFFER,
                  8,
                  true),
              new AccelerationProvider.Shape(
                  4,
                  1,
                  368,
                  Set.of(AccelerationProvider.Encoding.I64),
                  AccelerationProvider.LayoutKind.NATIVE_BUFFER,
                  8,
                  true),
              new AccelerationProvider.Shape(
                  4,
                  0,
                  369,
                  Set.of(AccelerationProvider.Encoding.I64),
                  AccelerationProvider.LayoutKind.NATIVE_BUFFER,
                  8,
                  true),
              new AccelerationProvider.Shape(
                  4,
                  0,
                  368,
                  Set.of(AccelerationProvider.Encoding.I64),
                  AccelerationProvider.LayoutKind.HEAP_BUFFER,
                  1,
                  false),
              new AccelerationProvider.Shape(
                  4,
                  0,
                  368,
                  Set.of(AccelerationProvider.Encoding.I64),
                  AccelerationProvider.LayoutKind.MAPPED_BUFFER,
                  8,
                  true))) {
        assertFalse(rejected.toString(), runtime.supports(capability, rejected, -10, 10, point));
      }
      assertTrue(
          runtime.supports(
              capability,
              new AccelerationProvider.Shape(
                  8,
                  0,
                  408,
                  Set.of(AccelerationProvider.Encoding.I64),
                  AccelerationProvider.LayoutKind.NATIVE_BUFFER,
                  8,
                  true),
              -20,
              20,
              frange));
      var after = runtime.inventory();
      assertEquals(
          "preflight does not count calls or fallbacks", before.counters(), after.counters());
      assertEquals(
          "preflight does not reserve or materialize storage", before.memory(), after.memory());
      assertEquals(0L, after.counters().get(capability).nativeCalls());
    } finally {
      if (previous == null) System.clearProperty(key);
      else System.setProperty(key, previous);
      if (previousState == null) System.clearProperty(stateKey);
      else System.setProperty(stateKey, previousState);
    }
  }

  public void testMappedBytePreflightUsesPhysicalNativeDescriptorAndMappedPolicy()
      throws Exception {
    Path pack = TestAccelerationRuntime.nativePack();
    Path file = TrustedPathFixtures.file("mapped-byte-preflight-policy");
    String row =
        AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + PackLoader.verify(pack).digest()
            + "\tx86-avx2-os-xsave-ymm\t"
            + Capability.VECTOR_I8.id
            + "\t384\t512\t512\t1\t-\tmapped\t199360\tTEST-ONLY-byte-preflight\n";
    String key = "lucene.acceleration.native.qualificationPath";
    String old = System.getProperty(key);
    Files.writeString(file, row);
    Files.setPosixFilePermissions(file, PosixFilePermissions.fromString("rw-------"));
    System.setProperty(key, file.toString());
    var mapped =
        new AccelerationProvider.Shape(
            512,
            384,
            199360,
            Set.of(AccelerationProvider.Encoding.BYTES),
            AccelerationProvider.LayoutKind.MAPPED_BUFFER,
            1,
            true);
    var physical =
        new AccelerationProvider.Shape(
            512,
            384,
            199360,
            Set.of(AccelerationProvider.Encoding.BYTES),
            AccelerationProvider.LayoutKind.NATIVE_BUFFER,
            1,
            true);
    try {
      var config =
          TestAccelerationRuntime.config(AccelerationConfig.Mode.AUTO, pack, Set.of(), Set.of(), 1);
      try (var runtime = new AccelerationRuntime(config)) {
        var before = runtime.inventory();
        assertTrue(before.rejections().toString(), runtime.supports(Capability.VECTOR_I8));
        assertTrue(
            "mapped qualification must use the native physical descriptor",
            runtime.supports(Capability.VECTOR_I8, mapped, 1, 0, "generic"));
        assertFalse(
            "mapped enrollment must not enroll owned native input",
            runtime.supports(Capability.VECTOR_I8, physical, 1, 0, "generic"));
        assertFalse(runtime.supports(Capability.VECTOR_I8, mapped, 0, 0, "generic"));
        var after = runtime.inventory();
        assertEquals(before.counters(), after.counters());
        assertEquals(before.memory(), after.memory());
      }
      Files.writeString(file, row.replace("\tmapped\t", "\tnative\t"));
      try (var runtime = new AccelerationRuntime(config)) {
        assertTrue(runtime.supports(Capability.VECTOR_I8, physical, 1, 0, "generic"));
        assertFalse(
            "native-only policy cannot authorize a mapped reader view",
            runtime.supports(Capability.VECTOR_I8, mapped, 1, 0, "generic"));
      }
    } finally {
      if (old == null) System.clearProperty(key);
      else System.setProperty(key, old);
    }
  }

  public void testProductionConsumerContextDeclinesBeforeNativeAndKeepsLegacyAggregate()
      throws Exception {
    Path pack = TestAccelerationRuntime.nativePack();
    Path file = TrustedPathFixtures.file("consumer-policy");
    String digest = PackLoader.verify(pack).digest();
    String context = "solr.point.uncached.full-live.v1";
    String prefix = AccelerationRuntime.BRIDGE_BUILD + "\t" + digest + "\tx86-avx2-os-xsave-ymm\t";
    Files.writeString(
        file,
        prefix
            + Capability.DOCVALUES_RANGE_FILTER.id
            + "\t0\t4\t4\t-10\t10\tnative\t8192\tunit-consumer-context\t"
            + context
            + "\n"
            + prefix
            + Capability.DOCVALUES_AGGREGATE.id
            + "\t64\t4\t4\t-10\t10\tnative\t8192\tunit-legacy-aggregate\n");
    Files.setPosixFilePermissions(file, PosixFilePermissions.fromString("rw-------"));
    Path state = TrustedPathFixtures.directory("consumer-state");
    String previousPolicy = System.getProperty("lucene.acceleration.native.qualificationPath");
    String previousState = System.getProperty("lucene.acceleration.native.statePath");
    System.setProperty("lucene.acceleration.native.qualificationPath", file.toString());
    System.setProperty("lucene.acceleration.native.statePath", state.toString());
    try {
      var config =
          TestAccelerationRuntime.config(AccelerationConfig.Mode.AUTO, pack, Set.of(), Set.of(), 1);
      try (var runtime = new AccelerationRuntime(config);
          var scope = runtime.newScope()) {
        assertFalse(runtime.isQualificationOnly());
        assertTrue(
            runtime.inventory().rejections().toString(),
            runtime.supports(Capability.DOCVALUES_RANGE_FILTER));
        assertTrue(
            runtime.inventory().rejections().toString(),
            runtime.supports(Capability.DOCVALUES_AGGREGATE));
        assertTrue(runtime.supports(Capability.DOCVALUES_RANGE_FILTER, context));
        assertFalse(runtime.supports(Capability.DOCVALUES_RANGE_FILTER, "generic"));
        assertFalse(
            runtime.supports(Capability.DOCVALUES_RANGE_FILTER, "solr.frange.full-live.v1"));
        assertFalse(runtime.supports(Capability.DOCVALUES_RANGE_FILTER, "bad context"));
        assertTrue(runtime.supports(Capability.DOCVALUES_AGGREGATE, "generic"));
        assertFalse(runtime.supports(Capability.DOCVALUES_AGGREGATE, context));
        assertEquals(
            "preflight never enters C",
            0,
            runtime.inventory().counters().get(Capability.DOCVALUES_RANGE_FILTER).nativeCalls());
        ByteBuffer values = scope.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
        values.putLong(0, -3).putLong(8, 0).putLong(16, 5).putLong(24, 9);
        ByteBuffer present = scope.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(0, 15);
        ByteBuffer parameters =
            scope
                .allocate(64)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(0, 4)
                .putLong(24, -10)
                .putLong(32, 10);
        ByteBuffer output = scope.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(0, 12345);
        ByteBuffer metadata = scope.allocate(256);
        ByteBuffer scratch = scope.allocate(0);
        var range =
            new AccelerationProvider.Batch(
                Capability.DOCVALUES_RANGE_FILTER,
                List.of(
                    new AccelerationProvider.Input(values, 4, AccelerationProvider.Encoding.I64),
                    new AccelerationProvider.Input(present, 1, AccelerationProvider.Encoding.I64)),
                new AccelerationProvider.Output(output, 1),
                parameters,
                scratch,
                metadata);
        assertFalse(
            "legacy generic API cannot use a point-specific profile",
            runtime.execute(range, 4, 0, scope));
        assertFalse(runtime.execute(range, 4, 0, scope, "solr.frange.full-live.v1"));
        assertEquals(
            "declined contexts cannot modify the caller's output", 12345, output.getLong(0));
        assertEquals(
            0, runtime.inventory().counters().get(Capability.DOCVALUES_RANGE_FILTER).nativeCalls());
        expectThrows(
            IllegalArgumentException.class,
            () -> runtime.execute(range, 4, 0, scope, "bad context"));
        assertEquals(
            0, runtime.inventory().counters().get(Capability.DOCVALUES_RANGE_FILTER).nativeCalls());
        assertTrue(runtime.execute(range, 4, 0, scope, context));
        assertEquals(15, output.getLong(0));
        assertEquals(
            1, runtime.inventory().counters().get(Capability.DOCVALUES_RANGE_FILTER).nativeCalls());
        assertFalse(
            "executing the tagged call does not enroll future generic calls",
            runtime.execute(range, 4, 0, scope));
        assertEquals(
            1, runtime.inventory().counters().get(Capability.DOCVALUES_RANGE_FILTER).nativeCalls());

        parameters.putLong(8, 64).putLong(16, 8).putLong(40, 1);
        ByteBuffer aggregateOutput = scope.allocate(48).order(ByteOrder.LITTLE_ENDIAN);
        var aggregate =
            new AccelerationProvider.Batch(
                Capability.DOCVALUES_AGGREGATE,
                List.of(
                    new AccelerationProvider.Input(values, 32, AccelerationProvider.Encoding.BYTES),
                    new AccelerationProvider.Input(present, 1, AccelerationProvider.Encoding.I64)),
                new AccelerationProvider.Output(aggregateOutput, 6),
                parameters,
                scratch,
                metadata);
        assertTrue(
            "the original 12-column aggregate policy remains usable via the original API",
            runtime.execute(aggregate, 4, 0, scope));
        assertEquals(4, aggregateOutput.getLong(0));
        assertEquals(11, aggregateOutput.getLong(8));
        assertEquals(-3, aggregateOutput.getLong(16));
        assertEquals(9, aggregateOutput.getLong(24));
        assertEquals(1, aggregateOutput.getLong(32));
        assertEquals(0, aggregateOutput.getLong(40));
        assertEquals(
            1, runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE).nativeCalls());
      }
    } finally {
      if (previousPolicy == null)
        System.clearProperty("lucene.acceleration.native.qualificationPath");
      else System.setProperty("lucene.acceleration.native.qualificationPath", previousPolicy);
      if (previousState == null) System.clearProperty("lucene.acceleration.native.statePath");
      else System.setProperty("lucene.acceleration.native.statePath", previousState);
    }
  }

  public void testNativeQualificationCannotEnrollMappedInputs() throws Exception {
    var shape =
        new AccelerationProvider.Shape(
            32,
            384,
            1000,
            Set.of(AccelerationProvider.Encoding.BYTES),
            AccelerationProvider.LayoutKind.NATIVE_BUFFER,
            1,
            true);
    for (String layout : List.of("native", "buffer", "mapped")) {
      var policy = QualificationPolicy.parse(row().replace("\tbuffer\t", "\t" + layout + "\t"));
      var profile =
          policy
              .forKernel("a".repeat(64), "x86-avx2-os-xsave-ymm", Capability.OSQ_GATHER)
              .getFirst();
      assertEquals(layout, layout.equals("mapped"), profile.shape(shape, true));
      assertEquals(layout, !layout.equals("mapped"), profile.shape(shape, false));
    }
  }

  public void testProductionUsesActualBorrowedBytesAndRecordsStartupQuarantine() throws Exception {
    Path pack = TestAccelerationRuntime.nativePack();
    Path file = TrustedPathFixtures.file("native-policy");
    String digest = PackLoader.verify(pack).digest();
    Files.writeString(
        file,
        AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + digest
            + "\tx86-avx2-os-xsave-ymm\t"
            + Capability.DOCSET_CARDINALITY.id
            + "\t0\t4\t4\t-\t-\tnative\t360\tunit-shape-admission\n"
            + AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + digest
            + "\tx86-avx2-os-xsave-ymm\t"
            + Capability.DOCSET_INTERSECT_COUNT.id
            + "\t0\t4\t4\t-\t-\tnative\t392\tunit-shape-admission\n");
    Files.setPosixFilePermissions(file, PosixFilePermissions.fromString("rw-------"));
    Path state = TrustedPathFixtures.directory("production-state");
    Files.setPosixFilePermissions(state, PosixFilePermissions.fromString("rwx------"));
    String previousPolicy = System.getProperty("lucene.acceleration.native.qualificationPath");
    String previousState = System.getProperty("lucene.acceleration.native.statePath");
    System.setProperty("lucene.acceleration.native.qualificationPath", file.toString());
    System.setProperty("lucene.acceleration.native.statePath", state.toString());
    try {
      var config =
          TestAccelerationRuntime.config(AccelerationConfig.Mode.AUTO, pack, Set.of(), Set.of(), 1);
      String identity;
      try (var runtime = new AccelerationRuntime(config);
          var scope = runtime.newScope()) {
        assertTrue(
            runtime.inventory().rejections().toString(),
            runtime.supports(Capability.DOCSET_CARDINALITY));
        identity = runtime.inventory().selected().get(Capability.DOCSET_CARDINALITY);
        ByteBuffer input = scope.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 4; i++) input.putLong(i * 8, -1);
        ByteBuffer parameters =
            scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN).putLong(0, 4).putLong(16, 256);
        ByteBuffer output = scope.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer metadata = scope.allocate(256);
        var batch =
            new AccelerationProvider.Batch(
                Capability.DOCSET_CARDINALITY,
                List.of(
                    new AccelerationProvider.Input(input, 4, AccelerationProvider.Encoding.I64)),
                new AccelerationProvider.Output(output, 1),
                parameters,
                scope.allocate(0),
                metadata);
        assertTrue(runtime.execute(batch, 4, 0, scope));
        assertEquals(256L, output.getLong(0));
        long calls =
            runtime.inventory().counters().get(Capability.DOCSET_CARDINALITY).nativeCalls();
        var larger =
            new AccelerationProvider.Batch(
                Capability.DOCSET_CARDINALITY,
                List.of(
                    new AccelerationProvider.Input(
                        scope.allocate(40), 5, AccelerationProvider.Encoding.I64)),
                new AccelerationProvider.Output(output, 1),
                parameters,
                scope.allocate(0),
                metadata);
        assertFalse(
            "stagedBytes zero must not hide a larger borrowed input",
            runtime.execute(larger, 4, 0, scope));
        assertEquals(
            calls, runtime.inventory().counters().get(Capability.DOCSET_CARDINALITY).nativeCalls());
      }
      try (var journal = new NativeCallJournal(state)) {
        journal.quarantine(NativeCallJournal.key(identity));
      }
      try (var restarted = new AccelerationRuntime(config)) {
        assertFalse(restarted.supports(Capability.DOCSET_CARDINALITY));
        assertTrue(
            "a sibling exact-integer kernel must remain independently selected",
            restarted.supports(Capability.DOCSET_INTERSECT_COUNT));
        assertFalse(
            restarted.inventory().counters().get(Capability.DOCSET_INTERSECT_COUNT).quarantined());
        assertTrue(
            restarted.inventory().counters().get(Capability.DOCSET_CARDINALITY).quarantined());
        assertFalse(restarted.eligible(Capability.DOCSET_CARDINALITY, 4));
        assertEquals(
            1L,
            restarted
                .inventory()
                .counters()
                .get(Capability.DOCSET_CARDINALITY)
                .fallbacks()
                .get(AccelerationRuntime.Reason.QUARANTINED)
                .longValue());
      }
      String startup = AccelerationRuntime.BRIDGE_BUILD + ":" + digest + ":pack-startup";
      try (var journal = new NativeCallJournal(state)) {
        int slot = journal.beginStartup(startup, digest);
        journal.end(slot);
        journal.quarantine(NativeCallJournal.key(startup));
      }
      try (var restarted = new AccelerationRuntime(config)) {
        assertFalse(restarted.hasKernels());
        assertTrue(
            restarted.inventory().rejections().toString(),
            restarted.inventory().rejections().stream()
                .anyMatch(label -> label.startsWith("PERSISTENT_PACK_QUARANTINE:")));
        assertTrue(
            restarted.inventory().counters().get(Capability.DOCSET_INTERSECT_COUNT).quarantined());
      }
    } finally {
      if (previousPolicy == null)
        System.clearProperty("lucene.acceleration.native.qualificationPath");
      else System.setProperty("lucene.acceleration.native.qualificationPath", previousPolicy);
      if (previousState == null) System.clearProperty("lucene.acceleration.native.statePath");
      else System.setProperty("lucene.acceleration.native.statePath", previousState);
    }
  }

  public void testMalformedOrPeerWritablePolicyRejected() throws Exception {
    expectThrows(
        java.io.IOException.class,
        () -> QualificationPolicy.parse(row().replace("384\t8", "384\t-1")));
    expectThrows(
        java.io.IOException.class,
        () ->
            QualificationPolicy.parse(row().replace(AccelerationRuntime.BRIDGE_BUILD, "obsolete")));
    expectThrows(
        java.io.IOException.class, () -> QualificationPolicy.parse(row() + "\textra\tunexpected"));
    Path policy = TrustedPathFixtures.file("policy");
    Files.writeString(policy, row());
    Files.setPosixFilePermissions(policy, PosixFilePermissions.fromString("rw-rw-r--"));
    expectThrows(java.io.IOException.class, () -> QualificationPolicy.read(policy));
    Files.setPosixFilePermissions(policy, PosixFilePermissions.fromString("rw-------"));
    assertEquals(1, QualificationPolicy.read(policy).profiles.size());
  }

  @org.apache.lucene.util.SuppressForbidden(reason = "ProcessBuilder redirection requires File")
  public void testInvalidImplicitPolicyDoesNotPoisonReadersOrWriters() throws Exception {
    TrustedPathFixtures.assumeHostAccess();
    Path output = Path.of(createTempFile("startup-probe", ".log").toString());
    String separator = System.getProperty("path.separator");
    String core =
        Path.of(
                AccelerationRuntime.class
                    .getProtectionDomain()
                    .getCodeSource()
                    .getLocation()
                    .toURI())
            .toString();
    String resources = core;
    var registrations =
        org.apache.lucene.codecs.Codec.class
            .getClassLoader()
            .getResources("META-INF/services/org.apache.lucene.codecs.Codec");
    while (registrations.hasMoreElements()) {
      var spi = registrations.nextElement();
      try (var stream = spi.openStream()) {
        String services =
            new String(stream.readNBytes(4096), java.nio.charset.StandardCharsets.UTF_8);
        if (services
            .lines()
            .anyMatch(
                line -> line.strip().equals("org.apache.lucene.codecs.lucene104.Lucene104Codec"))) {
          if (spi.getProtocol().equals("file"))
            resources = Path.of(spi.toURI()).getParent().getParent().getParent().toString();
          break;
        }
      }
    }
    String classpath =
        String.join(
            separator,
            core,
            resources,
            Path.of(
                    TestQualificationPolicy.class
                        .getProtectionDomain()
                        .getCodeSource()
                        .getLocation()
                        .toURI())
                .toString());
    var process =
        new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-cp",
                classpath,
                StartupProbe.class.getName())
            .redirectErrorStream(true)
            .redirectOutput(output.toFile())
            .start();
    if (!process.waitFor(30, java.util.concurrent.TimeUnit.SECONDS)) {
      process.destroyForcibly();
      fail("startup probe timed out; outcome unknown");
    }
    assertEquals(Files.readString(output), 0, process.exitValue());
    assertTrue(Files.readString(output), Files.readString(output).contains("STARTUP_PROBE_OK"));
  }

  public static class StartupProbe {
    public static void main(String[] ignored) throws Exception {
      System.setProperty("lucene.acceleration.native.mode", "typo");
      var runtime = AccelerationRuntime.system();
      if (runtime.hasKernels() || runtime.inventory().ready())
        throw new AssertionError("invalid startup must disable native readiness");
      try (var directory = new org.apache.lucene.store.ByteBuffersDirectory()) {
        try (var writer =
            new org.apache.lucene.index.IndexWriter(
                directory, new org.apache.lucene.index.IndexWriterConfig())) {
          var document = new org.apache.lucene.document.Document();
          document.add(
              new org.apache.lucene.document.StringField(
                  "id", "one", org.apache.lucene.document.Field.Store.NO));
          writer.addDocument(document);
        }
        try (var reader = org.apache.lucene.index.DirectoryReader.open(directory)) {
          if (reader.numDocs() != 1 || AccelerationRuntime.system() != runtime)
            throw new AssertionError("reader/writer startup changed registry");
        }
      }
      System.out.println("STARTUP_PROBE_OK");
    }
  }

  public void testImplicitStartupFailsSoftAndExplicitStartupRemainsStrict() throws Exception {
    TrustedPathFixtures.assumeHostAccess();
    for (String[] invalid :
        new String[][] {
          {"native.mode", "typo"},
          {"memory.nativeBytes", "NaN"},
          {"experimental.enabled", "yes"},
          {"native.mode", "required"}
        }) {
      String property = "lucene.acceleration." + invalid[0];
      String previous = System.getProperty(property);
      System.setProperty(property, invalid[1]);
      try {
        try (var runtime = AccelerationRuntime.initializeSystem()) {
          assertEquals(AccelerationRuntime.State.JAVA_ONLY, runtime.inventory().state());
          assertTrue(
              runtime.inventory().rejections().toString(),
              runtime.inventory().rejections().stream()
                  .anyMatch(s -> s.startsWith("STARTUP_DISABLED:")));
        }
        expectThrows(IllegalArgumentException.class, AccelerationConfig::fromSystemProperties);
      } finally {
        if (previous == null) System.clearProperty(property);
        else System.setProperty(property, previous);
      }
    }
    String mode = System.getProperty("lucene.acceleration.native.mode");
    String required = System.getProperty("lucene.acceleration.native.requiredCapabilities");
    System.setProperty("lucene.acceleration.native.mode", "required");
    System.setProperty("lucene.acceleration.native.requiredCapabilities", Capability.VECTOR_I8.id);
    try {
      expectThrows(
          java.io.IOException.class,
          () -> new AccelerationRuntime(AccelerationConfig.fromSystemProperties()));
      try (var runtime = AccelerationRuntime.initializeSystem()) {
        assertFalse(runtime.hasKernels());
      }
    } finally {
      if (mode == null) System.clearProperty("lucene.acceleration.native.mode");
      else System.setProperty("lucene.acceleration.native.mode", mode);
      if (required == null) System.clearProperty("lucene.acceleration.native.requiredCapabilities");
      else System.setProperty("lucene.acceleration.native.requiredCapabilities", required);
    }
  }
}
