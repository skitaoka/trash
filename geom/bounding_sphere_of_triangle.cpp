// 三角形にフィットする球を計算する。
float4 sphere(float3 const v1, float3 const v2, float3 const v3)
{
  float3 const a = v3 - v2;
  float3 const b = v1 - v3;
  float3 const c = v2 - v1;

  float3 center; // 球の中心座標
  float  radius; // 級の半径

  // 鈍角三角形か見ていく
  if (dot(a, b) >= 0) {
    // v3 が鈍角 c が最長辺
    center = v1 + c * 0.5f;
    radius = length(c);
  } else if (dot(b, c) >= 0) {
    // v1 が鈍角 a が最長辺
    center = v2 + a * 0.5f;
    radius = length(a);
  } else if (dot(c, a) >= 0) {
    // v2 が鈍角 b が最長辺
    center = v3 + b * 0.5f;
    radius = length(b);
  } else {
    // 鋭角三角形（外接球を求める）
    float const a2 = length_squared(a);
    float const b2 = length_squared(b);
    float const c2 = length_squared(c);
    center = (
        v1 * (a2 * (b2 + c2 - a2)) +
        v2 * (b2 * (c2 + a2 - b2)) +
        v3 * (c2 * (a2 + b2 - c2))
       ) / (4.0f * length_squared(cross(b, c)));
    radius = length(v1 - center);
  }

  return float4(center, radius);
}
