/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Experimental accounting and ownership for explicitly enrolled managed memory. Reservations
 * precede allocation, shared leases pin retired values, and request workers keep backing storage
 * alive until they drain. Failed reclamation retains its charge. These contracts optionally enroll
 * existing Lucene query-cache builders and values, ordinal builders and packed maps, and physical
 * file mappings when the corresponding explicit heap or mapped budget is enabled. Native cache
 * storage requires selectiveOffHeap policy; current remains the default. Mapped capacity is virtual
 * address space, and enrolled byte totals are not a bound on process RSS.
 *
 * @lucene.experimental
 */
package org.apache.lucene.acceleration.memory;
