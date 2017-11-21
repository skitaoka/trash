//
// Generating Samples w/o Concentric Disc Map
//
int i; // the index of a sample
int N; // the number of samples
double const theta = i * ((1.0 - 1.0 / 1.61803398874989) * Math.PI * 2.0);
double const radius = std::sqrt((i + 0.5) / N);
double const u = radius * std::cos(theta);
double const v = radius * std::sin(theta);
/*
  1.61803398874989 は、黄金比φ = (1+Sqrt[5])/2
  1-1/φ = 3-Sqrt[5]
*/
