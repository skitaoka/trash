//
// 3Dc: 法線マップに特化した非可逆圧縮の仕組み
// cf. http://www.watch.impress.co.jp/game/docs/20040505/rx800.htm
//

// エンコード(擬似コード)
void encode(std::ostream & out, normalmap const & map)
{
  assert(map.width () % 4 == 0);
  assert(map.height() % 4 == 0);

  out
    << map.width ()
    << map.height();

  for (int raw = 0, height = map.height(); raw < height; raw += 4) {
    for (int col = 0, width = map.width(); col < width; col += 4) {
      float min_x =  std::numeric_limits<float>::infinity();
      float max_x = -std::numeric_limits<float>::infinity();
      float min_y =  std::numeric_limits<float>::infinity();
      float max_y = -std::numeric_limits<float>::infinity();

      // 4x4領域の最大値を求める．
      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
          float3 const data = map(col + x, raw + y);

          if (min_x > data.x) {
            min_x = data.x;
          } else if (max_x < data.x) {
            max_x = data.x;
          }

          if (min_y > data.y) {
            min_y = data.y;
          } else if (max_y < data.y) {
            max_y = data.y;
          }
        }
      }

      out
        << min_x << max_x
        << min_y << max_y;

      // 8bitへ量子化
      float const range_x_inv = ((1<<8)-1) / (max_x - min_x);
      float const range_y_inv = ((1<<8)-1) / (max_y - min_y);
      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
          float3 const data = map(col + x, raw + y);
          out
            << static_cast<int>((data.x - min_x) * range_x_inv)
            << static_cast<int>((data.y - min_y) * range_y_inv);
        }
      }
    }
  }

// デコード(擬似コード)
void decode(normalmap & map, std::istream & in)
{
  int width, height;
  in >> width >> height;

  assert(width  % 4 == 0);
  assert(height % 4 == 0);

  map.resize(width, height);

  for (int raw = 0; raw < height; raw += 4) {
    for (int col = 0; col < width; col += 4) {
      float min_x, max_x;
      float min_y, max_y;

      in
        >> min_x >> max_x
        >> min_y >> max_y;

      // 復元
      float const range_x = (max_x - min_x) / ((1<<8)-1);
      float const range_y = (max_y - min_y) / ((1<<8)-1);
      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
          int data_x;
          int data_y;
          in
            >> data_x
            >> data_y;

          float const x = min_x + range_x * data_x;
          float const y = min_y + range_y * data_y;
          float const z = std::sqrt(1.0f - x * x - y * y);

          map(col + x, raw + y) = float3(x, y, z);
        }
      }
    }
  }
}
