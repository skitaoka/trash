//
// Generating Samples w/o Concentric Disc Map
//
int i; // the index of a sample
int N; // the number of samples
double const theta = i * ((1.0 - 1.0 / 1.61803398874989) * Math.PI * 2.0);
double const radius = std::sqrt((i + 0.5) / N);
double const u = radius * std::cos(theta);
double const v = radius * std::sin(theta);
