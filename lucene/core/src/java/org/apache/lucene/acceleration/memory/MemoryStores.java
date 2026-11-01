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
package org.apache.lucene.acceleration.memory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * Factory for independently owned typed stores, including externally mapped storage. Allocation is
 * admitted before invoking an allocator. Views never expose backing buffers.
 *
 * @lucene.experimental
 */
public final class MemoryStores {
  private MemoryStores() {}

  private record Failed(
      MemoryLedger ledger, NativeAllocator.Allocation allocation, MemoryLedger.Charge charge) {}

  private static final ConcurrentHashMap<MemoryLedger.Charge, Failed> FAILED =
      new ConcurrentHashMap<>();

  /** Explicit recovery for standalone typed-store and allocator-contract unwind close failures. */
  public static void retryFailedReclamations(MemoryLedger ledger) {
    RuntimeException failure = null;
    for (Failed resource : FAILED.values()) {
      if (resource.ledger() != ledger || FAILED.remove(resource.charge(), resource) == false)
        continue;
      try {
        resource.allocation().close();
        resource.charge().releaseAfterDeallocation();
      } catch (RuntimeException exception) {
        FAILED.put(resource.charge(), resource);
        if (failure == null) failure = exception;
        else failure.addSuppressed(exception);
      } catch (Error fatal) {
        FAILED.put(resource.charge(), resource);
        throw fatal;
      }
    }
    try {
      ManagedMemory.retryFailedReclamations(ledger);
    } catch (RuntimeException exception) {
      if (failure == null) failure = exception;
      else failure.addSuppressed(exception);
    }
    if (failure != null) throw failure;
  }

  /** Allocates an integer store. Mapped storage must use the explicit mapped factory. */
  public static IntStore ints(
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      MemoryLedger.Kind kind,
      int length,
      NativeAllocator allocator,
      MemoryLedger.Purpose purpose) {
    return new Ints(allocate(ledger, owner, kind, length, Integer.BYTES, allocator, purpose));
  }

  /** Allocates a long store. */
  public static LongStore longs(
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      MemoryLedger.Kind kind,
      int length,
      NativeAllocator allocator,
      MemoryLedger.Purpose purpose) {
    return new Longs(allocate(ledger, owner, kind, length, Long.BYTES, allocator, purpose));
  }

  /** Allocates a float store. */
  public static FloatStore floats(
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      MemoryLedger.Kind kind,
      int length,
      NativeAllocator allocator,
      MemoryLedger.Purpose purpose) {
    return new Floats(allocate(ledger, owner, kind, length, Float.BYTES, allocator, purpose));
  }

  /** Enrolls a mapping already admitted by the caller; takes ownership of its physical close. */
  public static IntStore mappedInts(ByteBuffer buffer, MemoryLedger.Charge charge, Runnable unmap) {
    return new Ints(mapped(buffer, charge, unmap, Integer.BYTES));
  }

  /** Enrolls an admitted long mapping; takes ownership of its physical close. */
  public static LongStore mappedLongs(
      ByteBuffer buffer, MemoryLedger.Charge charge, Runnable unmap) {
    return new Longs(mapped(buffer, charge, unmap, Long.BYTES));
  }

  /** Enrolls an admitted float mapping; takes ownership of its physical close. */
  public static FloatStore mappedFloats(
      ByteBuffer buffer, MemoryLedger.Charge charge, Runnable unmap) {
    return new Floats(mapped(buffer, charge, unmap, Float.BYTES));
  }

  static MemoryLedger.Charge charge(IntStore store) {
    return ((Ints) store).backing.charge;
  }

  static MemoryLedger.Charge charge(LongStore store) {
    return ((Longs) store).backing.charge;
  }

  static Bytes bytes(
      MemoryLedger ledger, MemoryLedger.Owner owner, int length, MemoryLedger.Purpose purpose) {
    return new Bytes(allocate(ledger, owner, MemoryLedger.Kind.HEAP, length, 1, null, purpose));
  }

  static final class Bytes implements AutoCloseable {
    private final Backing backing;

    Bytes(Backing backing) {
      this.backing = backing;
    }

    int get(int index) {
      return Byte.toUnsignedInt(backing.read().get(backing.offset(index)));
    }

    void set(int index, int value) {
      backing.write().put(backing.offset(index), (byte) value);
    }

    void freeze() {
      backing.freeze();
    }

    long capacityBytes() {
      return backing.capacityBytes();
    }

    MemoryLedger.Charge charge() {
      return backing.charge;
    }

    @Override
    public void close() {
      backing.close();
    }
  }

  private static Backing mapped(
      ByteBuffer buffer, MemoryLedger.Charge charge, Runnable unmap, int width) {
    Objects.requireNonNull(buffer);
    Objects.requireNonNull(charge);
    Objects.requireNonNull(unmap);
    if (charge.snapshot().kind() != MemoryLedger.Kind.MAPPED
        || charge.snapshot().capacity() < buffer.remaining()
        || buffer.remaining() % width != 0)
      throw new IllegalArgumentException("invalid mapped charge or alignment");
    return new Backing(buffer.slice().order(ByteOrder.nativeOrder()), charge, unmap, width);
  }

  private static Backing allocate(
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      MemoryLedger.Kind kind,
      int length,
      int width,
      NativeAllocator allocator,
      MemoryLedger.Purpose purpose) {
    long bytes = MemoryLedger.bytes(length, width);
    if (bytes > Integer.MAX_VALUE || kind == MemoryLedger.Kind.MAPPED)
      throw new IllegalArgumentException("unsupported storage size or kind");
    long capacity =
        kind == MemoryLedger.Kind.HEAP
            ? Math.addExact(
                RamUsageEstimator.alignObjectSize(
                    Math.addExact(bytes, RamUsageEstimator.NUM_BYTES_ARRAY_HEADER)),
                128)
            : Objects.requireNonNull(allocator).allocationSize(bytes);
    if (capacity < bytes) throw new IllegalArgumentException("allocator underestimates capacity");
    MemoryLedger.Admission admission = ledger.tryReserve(owner, kind, capacity, purpose);
    if (!admission.granted())
      throw new RequestMemoryScope.AdmissionException(
          RequestMemoryScope.Failure.valueOf(admission.reason().name()));
    try (MemoryLedger.Reservation reservation = admission.reservation()) {
      MemoryLedger.Charge charge = reservation.commit(capacity, MemoryLedger.State.BUILDING);
      NativeAllocator.Allocation allocation = null;
      try {
        ByteBuffer buffer;
        Runnable destroy;
        if (kind == MemoryLedger.Kind.HEAP) {
          buffer = ByteBuffer.allocate((int) bytes);
          destroy = () -> {};
        } else {
          allocation = allocator.allocate(bytes);
          if (allocation.capacity() > capacity || allocation.capacity() < bytes)
            throw new IllegalStateException("allocation exceeds admitted capacity");
          buffer = allocation.buffer().duplicate().clear().limit((int) bytes).slice();
          destroy = allocation::close;
        }
        return new Backing(buffer.order(ByteOrder.nativeOrder()), charge, destroy, width);
      } catch (RuntimeException | Error failure) {
        if (allocation != null) {
          try {
            allocation.close();
          } catch (RuntimeException | Error closeFailure) {
            FAILED.put(charge, new Failed(ledger, allocation, charge));
            charge.reclassify(MemoryLedger.State.FAILED_RECLAMATION);
            failure.addSuppressed(closeFailure);
            throw failure;
          }
        }
        charge.releaseAfterDeallocation();
        throw failure;
      }
    }
  }

  private static final class Backing {
    private ByteBuffer buffer;
    private final MemoryLedger.Charge charge;
    private final Runnable destroy;
    private final int width;
    private volatile boolean frozen;

    Backing(ByteBuffer buffer, MemoryLedger.Charge charge, Runnable destroy, int width) {
      this.buffer = buffer;
      this.charge = charge;
      this.destroy = destroy;
      this.width = width;
    }

    int length() {
      return read().capacity() / width;
    }

    ByteBuffer read() {
      ByteBuffer current = buffer;
      if (current == null) throw new IllegalStateException("closed typed store");
      return current;
    }

    ByteBuffer write() {
      if (frozen) throw new IllegalStateException("immutable typed store");
      return read();
    }

    int offset(int index) {
      Objects.checkIndex(index, length());
      return index * width;
    }

    void freeze() {
      read();
      frozen = true;
      charge.reclassify(MemoryLedger.State.PUBLISHED);
    }

    synchronized void close() {
      if (buffer == null) return;
      try {
        destroy.run();
      } catch (RuntimeException | Error failure) {
        charge.reclassify(MemoryLedger.State.FAILED_RECLAMATION);
        ManagedMemory.retainFailedReclamation(charge, this, this::close);
        throw failure;
      }
      buffer = null;
      charge.releaseAfterDeallocation();
      ManagedMemory.forgetFailedReclamation(charge, this);
    }

    long capacityBytes() {
      return charge.snapshot().capacity();
    }

    MemoryLedger.Kind kind() {
      return charge.snapshot().kind();
    }
  }

  private record Ints(Backing backing) implements IntStore {
    @Override
    public int length() {
      return backing.length();
    }

    @Override
    public int get(int index) {
      return backing.read().getInt(backing.offset(index));
    }

    @Override
    public void set(int index, int value) {
      backing.write().putInt(backing.offset(index), value);
    }

    @Override
    public IntStore freeze() {
      backing.freeze();
      return this;
    }

    @Override
    public long capacityBytes() {
      return backing.capacityBytes();
    }

    @Override
    public MemoryLedger.Kind kind() {
      return backing.kind();
    }

    @Override
    public void close() {
      backing.close();
    }
  }

  private record Longs(Backing backing) implements LongStore {
    @Override
    public int length() {
      return backing.length();
    }

    @Override
    public long get(int index) {
      return backing.read().getLong(backing.offset(index));
    }

    @Override
    public void set(int index, long value) {
      backing.write().putLong(backing.offset(index), value);
    }

    @Override
    public LongStore freeze() {
      backing.freeze();
      return this;
    }

    @Override
    public long capacityBytes() {
      return backing.capacityBytes();
    }

    @Override
    public MemoryLedger.Kind kind() {
      return backing.kind();
    }

    @Override
    public void close() {
      backing.close();
    }
  }

  private record Floats(Backing backing) implements FloatStore {
    @Override
    public int length() {
      return backing.length();
    }

    @Override
    public float get(int index) {
      return backing.read().getFloat(backing.offset(index));
    }

    @Override
    public void set(int index, float value) {
      backing.write().putFloat(backing.offset(index), value);
    }

    @Override
    public FloatStore freeze() {
      backing.freeze();
      return this;
    }

    @Override
    public long capacityBytes() {
      return backing.capacityBytes();
    }

    @Override
    public MemoryLedger.Kind kind() {
      return backing.kind();
    }

    @Override
    public void close() {
      backing.close();
    }
  }
}
