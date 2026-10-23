// Copyright (c) Meta Platforms, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "Io.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Pair.h>
#include <Corrade/Containers/StringStl.h>
#include <Corrade/Utility/FormatStl.h>
#include <Corrade/Utility/Path.h>
#ifdef _WIN32
#include <algorithm>
#include <filesystem>
#include <system_error>
#else
#include <glob.h>
#endif

namespace Cr = Corrade;
namespace esp {
namespace io {

std::string changeExtension(const std::string& filename,
                            const std::string& ext) {
  const Cr::Containers::StringView filenameBase =
      Cr::Utility::Path::splitExtension(filename).first();
  return Cr::Utility::formatString(
      ext.empty() || ext[0] == '.' ? "{}{}" : "{}.{}", filenameBase, ext);

}  // changeExtension

std::string normalizePath(const std::string& srcPath) {
  std::size_t ellipsisLoc = srcPath.find("/../");
  if ((ellipsisLoc == std::string::npos) || (ellipsisLoc == 0)) {
    // not found
    return srcPath;
  }
  // ellipsisLoc is location of first instance of "/../"
  // Get rid of ellipsis inside a specified path, and the previous directory
  // it references, so that if the ellisis spans a link, consumers don't get
  // lost.

  // Get end of path/filename after location of ellipsis
  auto suffixString = srcPath.substr(ellipsisLoc + 4);
  // Get beginning of path up to directory before ellipsis
  std::size_t prevLoc = srcPath.find_last_of('/', ellipsisLoc - 1);
  // Get paths leading up to ellipsis-cancelled path
  auto prefixString = srcPath.substr(0, prevLoc);
  // recurses to get subsequent ellipses.
  auto filteredPath =
      Cr::Utility::formatString("{}/{}", prefixString, suffixString);
  return normalizePath(filteredPath);
}  // normalizePath

std::string getPathRelativeToAbsPath(const std::string& toRelPath,
                                     const std::string& absPathArg) {
  // Check if absPath is a path or filename - only use the path if it is a
  // filenaame
  const std::string absPath = Cr::Utility::Path::path(absPathArg);

  std::string result = "";
  const char* delim = "/";

  Cr::Containers::Array<Cr::Containers::StringView>
      absDirs =
          Cr::Containers::StringView{absPath}.splitWithoutEmptyParts(delim[0]),
      relDirs = Cr::Containers::StringView{toRelPath}.splitWithoutEmptyParts(
          delim[0]);
  auto absIter = absDirs.cbegin();
  auto relIter = relDirs.cbegin();

  // find where both paths diverge - skip shared path components
  while (absIter != absDirs.cend() && relIter != relDirs.cend() &&
         *relIter == *absIter) {
    ++relIter;
    ++absIter;
  }

  // Add back-path components for each directory in abspath not found in
  // toRelPath
  while (absIter != absDirs.cend()) {
    Cr::Utility::formatInto(result, result.size(), "..{}", delim);

    ++absIter;
  }
  // Relative tail of path
  std::string relTail = "";
  while (relIter != relDirs.cend()) {
    if (*relIter == relDirs.back()) {
      Cr::Utility::formatInto(result, result.size(), "{}{}", relTail,
                              relDirs.back());
    } else {
      Cr::Utility::formatInto(relTail, relTail.size(), "{}{}", *relIter, delim);
    }
    ++relIter;
  }
  return result;
}  // getPathRelativeToAbsPath

#ifdef _WIN32
namespace {

// Minimal fnmatch() equivalent supporting '*', '?' and '[...]' (with '!' or
// '^' negation and ranges), matching a single path component.
bool wildcardMatch(const char* pat, const char* str) {
  while (*pat) {
    if (*pat == '*') {
      while (*pat == '*')
        ++pat;
      if (!*pat)
        return true;
      for (; *str; ++str)
        if (wildcardMatch(pat, str))
          return true;
      return false;
    }
    if (!*str)
      return false;
    if (*pat == '?') {
      ++pat;
      ++str;
    } else if (*pat == '[') {
      const char* p = pat + 1;
      bool negate = (*p == '!' || *p == '^');
      if (negate)
        ++p;
      bool matched = false;
      bool first = true;
      while (*p && (first || *p != ']')) {
        first = false;
        char lo = *p;
        char hi = lo;
        if (p[1] == '-' && p[2] && p[2] != ']') {
          hi = p[2];
          p += 3;
        } else {
          ++p;
        }
        if (*str >= lo && *str <= hi)
          matched = true;
      }
      if (*p != ']') {
        // unterminated bracket: treat '[' literally
        if (*str != '[')
          return false;
        ++pat;
        ++str;
        continue;
      }
      if (matched == negate)
        return false;
      pat = p + 1;
      ++str;
    } else {
      if (*pat != *str)
        return false;
      ++pat;
      ++str;
    }
  }
  return *str == '\0';
}

bool hasWildcard(const std::string& s) {
  return s.find_first_of("*?[") != std::string::npos;
}

}  // namespace

std::vector<std::string> globDirs(const std::string& pattern) {
  namespace fs = std::filesystem;
  // Normalize separators to '/' to match the rest of the codebase.
  std::string pat = pattern;
  std::replace(pat.begin(), pat.end(), '\\', '/');

  // Split off an absolute prefix ("C:/", "/" or "//server/share/").
  std::string prefix;
  std::size_t start = 0;
  if (pat.size() >= 2 && pat[1] == ':') {
    prefix = pat.substr(0, 2);
    start = 2;
    if (pat.size() > 2 && pat[2] == '/') {
      prefix += '/';
      start = 3;
    }
  } else if (!pat.empty() && pat[0] == '/') {
    prefix = "/";
    start = 1;
  }

  std::vector<std::string> components;
  for (std::size_t pos = start; pos <= pat.size();) {
    std::size_t next = pat.find('/', pos);
    if (next == std::string::npos)
      next = pat.size();
    if (next > pos)
      components.push_back(pat.substr(pos, next - pos));
    pos = next + 1;
  }

  std::vector<std::string> current{prefix};
  for (const std::string& comp : components) {
    std::vector<std::string> nextPaths;
    for (const std::string& base : current) {
      const std::string sep =
          (base.empty() || base.back() == '/') ? std::string{} : "/";
      if (!hasWildcard(comp)) {
        nextPaths.push_back(base + sep + comp);
        continue;
      }
      std::error_code ec;
      const fs::path dir = base.empty() ? fs::path{"."} : fs::path{base};
      for (fs::directory_iterator it{dir, ec}, end; !ec && it != end;
           it.increment(ec)) {
        const std::string name = it->path().filename().string();
        // Like POSIX glob, wildcards don't match a leading '.'
        if (!name.empty() && name[0] == '.' && comp[0] != '.')
          continue;
        if (wildcardMatch(comp.c_str(), name.c_str()))
          nextPaths.push_back(base + sep + name);
      }
    }
    current = std::move(nextPaths);
  }

  std::vector<std::string> ret;
  for (const std::string& p : current) {
    std::error_code ec;
    if (p.empty() || !fs::exists(p, ec))
      continue;
    // Emulate GLOB_MARK: append a slash to directories
    if (fs::is_directory(p, ec) && p.back() != '/')
      ret.push_back(p + "/");
    else
      ret.push_back(p);
  }
  std::sort(ret.begin(), ret.end());
  return ret;
}
#else
std::vector<std::string> globDirs(const std::string& pattern) {
  // Check for ellipsis, if so process here.
  glob_t glob_result;

  glob(pattern.c_str(), GLOB_MARK, nullptr, &glob_result);
  std::vector<std::string> ret(glob_result.gl_pathc);
  for (int i = 0; i < glob_result.gl_pathc; ++i) {
    ret[i] = std::string(glob_result.gl_pathv[i]);
  }
  globfree(&glob_result);
  return ret;
}
#endif

}  // namespace io
}  // namespace esp
