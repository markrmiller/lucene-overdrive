/* Licensed to the Apache Software Foundation (ASF) under one or more
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Small strict manifest parser with byte, depth, entry and string bounds. */
final class BoundedJson {
  private final String text;
  private int position;
  private int entries;

  private BoundedJson(String text) {
    this.text = text;
  }

  static Object parse(String text) throws IOException {
    if (text.length() > 65536) throw new IOException("manifest too large");
    BoundedJson parser = new BoundedJson(text);
    Object value = parser.value(0);
    parser.space();
    if (parser.position != text.length()) throw new IOException("trailing manifest data");
    return value;
  }

  private Object value(int depth) throws IOException {
    space();
    if (depth > 16 || ++entries > 2048 || position == text.length())
      throw new IOException("manifest bounds");
    char c = text.charAt(position);
    if (c == '"') return string();
    if (c == '{') {
      position++;
      Map<String, Object> result = new LinkedHashMap<>();
      if (consume('}')) return result;
      do {
        space();
        if (position == text.length() || text.charAt(position) != '"')
          throw new IOException("manifest object key");
        String key = string();
        if (!consume(':') || result.containsKey(key))
          throw new IOException("duplicate/invalid key");
        result.put(key, value(depth + 1));
      } while (consume(','));
      if (!consume('}')) throw new IOException("manifest object terminator");
      return result;
    }
    if (c == '[') {
      position++;
      List<Object> result = new ArrayList<>();
      if (consume(']')) return result;
      do {
        result.add(value(depth + 1));
      } while (consume(','));
      if (!consume(']')) throw new IOException("manifest array terminator");
      return result;
    }
    for (String literal : List.of("true", "false", "null")) {
      if (text.startsWith(literal, position)) {
        position += literal.length();
        return switch (literal) {
          case "true" -> true;
          case "false" -> false;
          default -> null;
        };
      }
    }
    int start = position;
    if (c == '-') position++;
    if (position == text.length() || !digit(text.charAt(position)))
      throw new IOException("manifest value");
    if (text.charAt(position) == '0') position++;
    else while (position < text.length() && digit(text.charAt(position))) position++;
    try {
      return Long.valueOf(text.substring(start, position));
    } catch (NumberFormatException e) {
      throw new IOException("manifest integer", e);
    }
  }

  private String string() throws IOException {
    position++;
    StringBuilder value = new StringBuilder();
    while (position < text.length()) {
      char c = text.charAt(position++);
      if (c == '"') return value.toString();
      if (c < 32 || value.length() >= 4096) throw new IOException("manifest string bounds");
      if (c == '\\') {
        if (position == text.length()) throw new IOException("manifest escape");
        c = text.charAt(position++);
        c =
            switch (c) {
              case '"', '\\', '/' -> c;
              case 'b' -> '\b';
              case 'f' -> '\f';
              case 'n' -> '\n';
              case 'r' -> '\r';
              case 't' -> '\t';
              case 'u' -> {
                if (text.length() - position < 4) throw new IOException("unicode escape");
                for (int i = position; i < position + 4; i++) {
                  char hex = text.charAt(i);
                  if (!digit(hex) && !(hex >= 'a' && hex <= 'f') && !(hex >= 'A' && hex <= 'F'))
                    throw new IOException("unicode escape");
                }
                try {
                  char decoded =
                      (char) Integer.parseInt(text.substring(position, position + 4), 16);
                  position += 4;
                  yield decoded;
                } catch (NumberFormatException e) {
                  throw new IOException("unicode escape", e);
                }
              }
              default -> throw new IOException("manifest escape");
            };
      }
      value.append(c);
    }
    throw new IOException("unterminated manifest string");
  }

  private boolean consume(char c) {
    space();
    if (position < text.length() && text.charAt(position) == c) {
      position++;
      return true;
    }
    return false;
  }

  private void space() {
    while (position < text.length() && " \n\r\t".indexOf(text.charAt(position)) >= 0) position++;
  }

  private static boolean digit(char c) {
    return c >= '0' && c <= '9';
  }
}
