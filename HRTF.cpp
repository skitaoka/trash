#include <cstdio>
#include <cstdlib>

#include <string>
#include <vector>


#include <iostream>
#include <sstream>
#include <fstream>

#define DATA_SIZE 512

namespace file
{
  bool exists(char const * const file)
  {
    FILE * fp;
    if (::fopen_s(&fp, file, "r")) {
      return false;
    }
    std::fclose(fp);
    return true;
  }

  bool exists(std::string const & file)
  {
    FILE * fp;
    if (::fopen_s(&fp, file.c_str(), "r")) {
      return false;
    }
    std::fclose(fp);
    return true;
  }

  void split(std::string const & path, std::string * const dir, std::string * const fname)
  {
    std::size_t const npos = path.find_last_of('\\');
    if (std::string::npos == npos) {
      *fname = path;
    } else {
      std::size_t const length = path.length();
      *dir = path.substr(0, npos+1);
      *fname = path.substr(npos+1);
    }
  }

}//end of namespace file

int main(int argc, char* argv[])
{
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s channel={L,R} distance={20,50}, elevation\n", argv[0]);
    return 1;
  }

  char const channel = argv[1][0];
  int const distance = std::atoi(argv[2]);
  int const elevation = std::atoi(argv[3]);

  char filename[_MAX_PATH];

  // ファイルの存在チェック
  for (int azimuth = 0; azimuth < 360; azimuth += 5) {
    ::sprintf_s(filename, "%c%dd%de%03da.dat", channel, distance, elevation, azimuth);
    if (!file::exists(filename)) {
      std::fprintf(stderr, "file not found (%d)! %s\n", __LINE__, filename);
      return 1;
    }
  }

  // 出力先のファイルを用意
  ::sprintf_s(filename, "%c%dd%de.csv", channel, distance, elevation);
  if (file::exists(filename)) {
    if (std::remove(filename)) {
      std::fprintf(stderr, "file not found (%d)! %s\n", __LINE__, filename);
      return 1;
    }
  }

  // 出力ファイルを開く
  std::ofstream out(filename);
  if (!out) {
    std::fprintf(stderr, "file not found (%d)! %s\n", __LINE__, filename);
    return 1;
  }

  for (std::size_t t = 0; t < DATA_SIZE/2; ++t) {
    out << ", " << t;
  }
  out << std::endl;
  for (int azimuth = 0; azimuth < 360; azimuth += 5) {
    // 入力ファイルを開く
    ::sprintf_s(filename, "%c%dd%de%03da.dat", channel, distance, elevation, azimuth);
    std::ifstream in(filename);
    if (!in) {
      std::fprintf(stderr, "file not found (%d)! %s\n", __LINE__, filename);
      return 1;
    }
    std::fprintf(stdout, "file processing ... %s\n", filename);

    // デーを読みつつ書き出し
    out << azimuth;
    for (std::size_t t = 0; t < DATA_SIZE/2; ++t) {
      double data;
      in >> data;
      out << ", " << data;
    }
    out << std::endl;
  }
}


#if 0
int main(int argc, char* argv[])
{
  std::vector<double> data(DATA_SIZE);
  for (int i = 1; i < argc; ++i) {
    std::ifstream in(argv[i]);
    if (!in) {
      std::cout << "file not found! " << argv[i] << std::endl;
      std::getchar();
      return 1;
    }

    for (std::size_t t = 0; t < DATA_SIZE; ++t) {
      in >> data[t];
    }

    for (std::size_t t = DATA_SIZE/4; t < DATA_SIZE; ++t) {
      if (0.0 != data[t]) {
        std::cout << "[!]" << std::endl;
        std::getchar();
        return 0;
      }
    }
  }

  return 0;
}
#endif

#if 0
int main(int argc, char* argv[])
{
  for (int i = 1; i < argc; ++i) {
    std::string oldname(argv[i]);

    //oldname が存在しなければ処理しない
    if (!file::exists(oldname)) {
      continue;
    }

    std::string dir, file;
    file::split(oldname, &dir, &file);
    
    char channel;   // {L,R}
    int  distance;  // [cm]
    char distance_symbol;
    int  elevation; // {-50, 90} [deg]
    char elevation_symbol;
    int  azimuth;   // {0, 360} [deg]
    char azimuth_symbol;
    {
      std::istringstream sin(file);
      sin >> channel
          >> distance
          >> distance_symbol
          >> elevation
          >> elevation_symbol
          >> azimuth
          >> azimuth_symbol;
      if ('d' != distance_symbol) {
        std::fprintf(stderr, "invalid distance symbol: \'%c\'\n", distance_symbol);
        continue;
      }
      if ('e' != elevation_symbol) {
        std::fprintf(stderr, "invalid elevation symbol: \'%c\'\n", elevation_symbol);
        continue;
      }
      if ('a' != azimuth_symbol) {
        std::fprintf(stderr, "invalid azimuth symbol: \'%c\'\n", azimuth_symbol);
        continue;
      }
    }

    if (('L' != channel) && ('R' != channel)) {
      std::fprintf(stderr, "invalid chanel: \'%c\'\n", channel);
      continue;
    }

    if ((distance < 0) || (999999 < distance)) { // 6 桁まで
      std::fprintf(stderr, "invalid distance: %d\n", distance);
      continue;
    }

    if ((elevation < -50) && (90 < elevation)) {
      std::fprintf(stderr, "invalid elevation: %d\n", elevation);
      continue;
    }

    if ((azimuth < 0) && (360 < azimuth)) {
      std::fprintf(stderr, "invalid azimuth: %d\n", azimuth);
      continue;
    }

    {
      char buffer[_MAX_PATH];
      ::sprintf_s(buffer, "%s%c%06dd%03de%03da.txt",
        dir.c_str(), channel, distance, elevation, azimuth);
      std::string newname(buffer);
      if (oldname != newname) {
        //newname が存在したら削除する
        if (file::exists(newname)) {
          std::remove(newname.c_str());
        }
        std::rename(oldname.c_str(), newname.c_str());
        std::cout << "\"" << oldname << "\" => " << newname << "\"" << std::endl;
      }
    }
  }

  return 0;
}
#endif
