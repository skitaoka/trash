#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// 成功したときに 0 が返る。
// 失敗したときは -1 が返る。
int remove_file(char const * const path)
{
#ifdef _WIN32
  return ::DeleteFile(path) ? 0 : -1;
#else
  return ::unlink(path);
#endif
}

// 成功したときに 0 が返る。
// 失敗したときは -1 が返る。
int rename_file(char const * const path_old, char const * const path_new)
{
#ifdef _WIN32
  return ::MoveFileEx(path_old, path_new, MOVEFILE_REPLACE_EXISTING) ? 0 : -1;
#else
  return ::rename(path_old, path_new);
#endif
}

int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s old_file_path new_file_path\n", argv[0]);
    return 1;
  }
  if (rename_file(argv[1], argv[2])) {
    std::fprintf(stderr, "failed to rename: \'%s\' to \'%s\'\n", argv[1], argv[2]);
  } else {
    std::fprintf(stderr, "succeeded to rename: \'%s\' to \'%s\'\n", argv[1], argv[2]);
  }

  return 0;
}
