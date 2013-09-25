/* 
 * Fast Ray-Box Intersection
 * by Andrew Woo
 * from "Graphics Gems", Academic Press, 1990
 *
 * 交差したときの交点の情報が必要で、
 * それ以外の情報を必要としない場合
 */

template < typename T >
bool RayBoxIntersection(
  tvmet::Vector<T, 3> const & min,
  tvmet::Vector<T, 3> const & max,
  tvmet::Vector<T, 3> const & origin,
  tvmet::Vector<T, 3> const & direction,
  tvmet::Vector<T, 3> const & intersection)
{
  tvmet::Vector<T, 3> candidatePlane;
  tvmet::Vector<bool, 3> quadrant;
  bool inside = true;

  // 光線の原点が箱に対してどの位置にあるかを判定する
  for (std::size_t i = 0; i < 3; i++) {
    if (origin[i] < min[i]) {
      candidatePlane[i] = min[i];    // この次元で最も近い平面の座標
      quadrant[i] = true;        // この次元で箱の領域外であるか
      inside = false;          // 箱の外部に存在する
    } else if (origin[i] > max[i]) {
      candidatePlane[i] = max[i];
      quadrant[i] = true;
      inside = false;
    } else {
      if (direction[i] < 0) {
        candidatePlane[i] = max[i];
      } else {
        candidatePlane[i] = min[i];
      }
      quadrant[i] = false;
    }
  }

  if (inside) {
    // 原点が箱の中にある場合
    intersection = origin;
  } else {
    // 原点が箱の外にある場合

    // 次元ごとの距離を出す
    Vector3 maxT;
    for (i = 0; i < 3; i++) {
      if (quadrant[i] && (dir[i] != 0.0)) {
        // 光線の原点がその次元で内部に存在していなくて
        // 光線の方向がそちらに向かっていれば
        // その次元での距離を出す
        maxT[i] = (candidatePlane[i] - origin[i]) / dir[i];
      } else {
        // 箱の内部に存在するかあるいは確実に交差しない
        maxT[i] = -1.0;
      }
    }

    // 最大の距離を持っている次元を探す（その次元での距離が交点までの距離と一致する）
    int whichPlane = 0;
    for (int i = 1; i < 3; ++i) {
      if (maxT[ whichPlane ] < maxT[i]) {
        whichPlane = i;
      }
    }

    // 光線が逆の方向を向いていて交差していない
    if (maxT[ whichPlane ] < 0.0) {
      return false;
    }

    for (int i = 0; i < 3; ++i) {
      if (whichPlane != i) {
        // この次元での交点の座標をだす
        intersection[i] = origin[i] + maxT[whichPlane] * dir[i];
        if ((intersection[i] < min[i]) || (intersection[i] > max[i])) {
          // この次元の交点が箱の領域に入っていなければアボーン
          return false;
        }
      } else {
        // 交差する次元はそのまま平面の座標を使う
        intersection[i] = candidatePlane[i];
      }
    }
  }

  return true;  // ray hits box
}
