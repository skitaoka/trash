/*
 *  Copyright (c) 2010 TOKUNAGA Hiroyuki
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1. Redistributions of source code must retain the above Copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above Copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. Neither the name of the authors nor the names of its contributors
 *      may be used to endorse or promote products derived from this
 *      software without specific prior written permission.
 *
 *   cf. ディレクトリの中にある大量の小さなファイルを高速に読み込む方法
 *       http://d.hatena.ne.jp/tkng/20090727/1248652900
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

typedef struct ifile_ {
  ino_t ino;
  char  *filename;
} ifile_t;

static char *construct_filename(const char *dir, const char *basename)
{
  size_t dlen = strlen(dir);
  size_t blen = strlen(basename);
  char *filename = malloc(dlen + blen + 2);
  strcpy(filename, dir);
  filename[dlen] = '/';
  strcpy(filename + dlen + 1, basename);
  return filename;
}


static void read_file(const char *dir_path, const char *filename)
{
  size_t bufsize = 32768;
  char *fullpath = construct_filename(dir_path, filename);
  char buf[bufsize];
  int ret;
  int fd = open(fullpath, O_RDONLY);

  if (fd < 0) {
      free(fullpath);
      return;
  }

  ret = read(fd, buf, bufsize);

  while (ret > 0 || (ret == -1 && errno == EINTR)) {
    ret = read(fd, buf, bufsize);
  }

  close(fd);
  free(fullpath);
}

int main(int argc, char **argv)
{
  bool sort_files = true;
  if (argc <= 1) {
    printf("dirdump: missing operand\n");
    return -1;
  } else if (argc == 3) {
    if (strcmp("--nosort", argv[2]) == 0) {
      sort_files = false;
    }
  }

  std::vector<ifile_t> ifiles;

  char const * const dir_path = argv[1];
  DIR * const dir = opendir(dir_path);
  __try {
  if (dir == NULL) {
      printf("open failed: %s\n", dir_path);
      return -1;
    }
    for (struct dirent * entry = readdir(dir); entry != NULL; entry = readdir(dir);
      ifiles.push_back(ifile_t());
      ifiles.back().ino = entry->d_ino;
      ifiles.back().filename = strdup(entry->d_name);
    }
  } __finally {
    closedir(dir);
  }

  if (sort_files) {
  	std::sort(ifiles.begin(), ifiles.end(),
	  [](ifile_t const & a, ifile_t const & b)
    {
      return a.ino - b.ino;
    });
  }

  for (std::size_t i = 0, size = ifiles.size(); i < size; ++i) {
    read_file(dir_path, ifiles[i].filename);
  }

  return 0;
}
