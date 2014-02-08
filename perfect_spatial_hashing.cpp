// 有効なデータ数を　n, ハッシュの次元数を d とすると
//   m = (1.01  * n)^(1/d) + 1
//   r = (sigma * n)^(1/d) + 1, sigma = 1/(2d)
// ぐらいで　m と r を決める.
bool build_perfect_spatial_hashing(
  image_t & image,    ///< original image
  image_t & hash,     ///< hash table
  image_t & offset,   ///< offset table
  int const u,        ///< original image size
  int const m,        ///< hash table size
  int const r)        ///< offset table size
{
  // テーブルのサイズをチェックする
  assert(u <= m * r);
  assert(gcd(m, r) == 1);

  // ハッシュテーブルをクリアする
  hash.fill(255);

  // オフセットテーブルをクリアする
  offset.fill(255);

  // max offset value.
  int const max_offset_value = std::min(m, 256);

  // Loop over sorted cardinals
  for (int y = 0; y < r; ++y) {
    for (int x = 0; x < r; ++x) {
      // Search all points of h^(-1)(x,y)
      std::vector<Point> points;
      for (int j = y; j < u; j += r) {
        for (int i = x; i < u; i += r) {
          if (non_white_pixel(image, i, j)) {
            points.push_back(Point(i, j));
          }
        }
      }
      if (points.empty()) {
        continue;
      }

      // Search for possible translations (the ones with no collisions)
      std::vector<Point> valid_translations;
      for (int offset_y = 0; offset_y < max_offset_value; ++offset_y) {
        for (int offset_x = 0; offset_x < max_offset_value; ++offset_x) {
          bool valid = true;
          for (unsigned int s = 0; s < points.size(); ++s) {
            Point const p = points[s];
            int const px = (p.x + offset_x) % m;
            int const py = (p.y + offset_y) % m;
            if (non_white_pixel(hash, px, py)) {
              valid = false;
              break;
            }
          }
          if (valid) {
            valid_translations.push_back(Point(offset_x, offset_y));
          }
        }
      }
      if (valid_translations.empty()) {
        return false;
      }

      { // Assign
        // Random selection of a valid translation
        Point const offset = valid_translations[rand() % valid_translations.size()];

        // Offset assignment
        offset(x, y, 0, 0) = offset.x;
        offset(x, y, 0, 1) = offset.y;

        // Hash table assignment
        for (unsigned int s = 0; s < points.size(); ++s) {
          Point const p = points[s];
          int const px = (p.x + offset.x) % m;
          int const py = (p.y + offset.y) % m;
          for (unsigned int k = 0; k < image.dim; ++k) {
            hash(px, py, k) = image(p.x, p.y, k);
          }
        }
      }
    }
  }

  return true;
}
