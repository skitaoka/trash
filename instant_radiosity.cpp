void InstantRadiosity( int N, double rho )
{
  double Start = N;

  for ( int Reflections = 0, End = N; End > 0; End = (int)Start, Reflections++ ) {
    Start *= rho;

    for ( int i = (int)Start; i < End; i++ ) {
      // Select starting point on light source
      Point y( phi( 2, i ), phi( 3, i ) );
      Color L( Le( y ) ); // Le( y ) * supp Le; supp Le?
      double w = N;

      // trace reflections
      for ( int j = 0; j <= Reflections; j++ ) {
        glRenderShadowedScene( N / floor( w ) * L, y );
        glAccum( GL_ACCUM, 1 / N );

        // diffuse scattering
        Vector ω = ωd( phi( 2 * j + 2, i ), phi( 2 * j + 3, ( i ) );

        //trace ray from y into direction ω
        y = h( y, ω );

        // Attenuate and compensate
        L *= fd( y );
        w *= ρ;
      }
    }
  }

  glAccum( GL_RETURN, 1.0 );
}
