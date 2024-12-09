
__constant double3 palette[16] = {
    (double3)(106.0 / 255.0, 52.0 / 255.0, 3.0 / 255.0),    // Brown 2 (ancien dernier)
    (double3)(153.0 / 255.0, 87.0 / 255.0, 0.0 / 255.0),    // Brown 1
    (double3)(204.0 / 255.0, 128.0 / 255.0, 0.0 / 255.0),   // Brown 0
    (double3)(255.0 / 255.0, 170.0 / 255.0, 0.0 / 255.0),   // Dirty yellow
    (double3)(248.0 / 255.0, 201.0 / 255.0, 95.0 / 255.0),  // Light yellow
    (double3)(241.0 / 255.0, 233.0 / 255.0, 191.0 / 255.0), // Lightest yellow
    (double3)(211.0 / 255.0, 236.0 / 255.0, 248.0 / 255.0), // Lightest blue
    (double3)(134.0 / 255.0, 181.0 / 255.0, 229.0 / 255.0), // Blue 0
    (double3)(57.0 / 255.0, 125.0 / 255.0, 209.0 / 255.0),  // Blue 1
    (double3)(24.0 / 255.0, 82.0 / 255.0, 177.0 / 255.0),   // Blue 2
    (double3)(12.0 / 255.0, 44.0 / 255.0, 138.0 / 255.0),   // Blue 3
    (double3)(0.0 / 255.0, 7.0 / 255.0, 100.0 / 255.0),     // Blue 4
    (double3)(4.0 / 255.0, 4.0 / 255.0, 73.0 / 255.0),      // Blue 5
    (double3)(9.0 / 255.0, 1.0 / 255.0, 47.0 / 255.0),      // Darkest blue
    (double3)(25.0 / 255.0, 7.0 / 255.0, 26.0 / 255.0),     // Dark violet
    (double3)(66.0 / 255.0, 30.0 / 255.0, 15.0 / 255.0)     // Brown 3 (ancien premier)
};



inline bool inside_cardioid(double cr, double ci) {
    double x_minus_quarter = cr - 0.25;
    double q = x_minus_quarter * x_minus_quarter + ci * ci;
    return (q * (q + x_minus_quarter) <= 0.25 * (ci * ci));
}

inline bool inside_bulb_period2(double cr, double ci) {
    double t = (cr + 1.0)*(cr + 1.0) + ci * ci;
    return (t <= 1.0/16.0);
}

inline bool inside_other_bulbs(double cr, double ci) {
    return false;
}

inline double3 getColor(double smooth_iter, double scale) {
    double tt = smooth_iter * scale;
    int colorIdx1 = ((int)floor(tt)) & 15;
    int colorIdx2 = (colorIdx1 + 1) & 15;
    double frac = tt - floor(tt);

    double3 color1 = palette[colorIdx1];
    double3 color2 = palette[colorIdx2];

    return color1 * (1.0 - frac) + color2 * frac;
}

inline double distanceEstimate(double Z_re, double Z_im, double dZ_re, double dZ_im) {
    double mag = sqrt(Z_re*Z_re + Z_im*Z_im);
    double dZ_mag = sqrt(dZ_re*dZ_re + dZ_im*dZ_im);
    return 2.0 * mag * log(mag) / dZ_mag; 
}

__kernel void mandelbrot(__global unsigned char* image, int width, int height,
                         double zoom, double offsetX, double offsetY, int maxIter, int samples) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int quickIter = maxIter / 6; 

    double half_width = (double)width * 0.5;
    double half_height = (double)height * 0.5;
    double base_c_re = (x - half_width) * zoom + offsetX;
    double base_c_im = (y - half_height) * zoom + offsetY;
    double delta = zoom / (double)samples;
    double inv_samples_sq = 1.0 / ((double)samples * (double)samples);

    double log_2 = log(2.0);
    double scale = 16.0 / 250.0;

    double color_r = 0.0;
    double color_g = 0.0;
    double color_b = 0.0;


    {
        double c_re_q = base_c_re;
        double c_im_q = base_c_im;

        if (inside_cardioid(c_re_q, c_im_q) || inside_bulb_period2(c_re_q, c_im_q) || inside_other_bulbs(c_re_q, c_im_q)) {
            int idx = (y * width + x) * 3;
            image[idx]     = 0;
            image[idx + 1] = 0;
            image[idx + 2] = 0;
            return;
        }

        // Quelques itérations rapides
        double Zr = c_re_q, Zi = c_im_q;
        double Zr2 = Zr*Zr, Zi2 = Zi*Zi;
        int qi = 0;
        for (; qi < quickIter && (Zr2 + Zi2 <= 4.0); qi++) {
            Zi = 2.0*Zr*Zi + c_im_q;
            Zr = Zr2 - Zi2 + c_re_q;
            Zr2 = Zr*Zr; 
            Zi2 = Zi*Zi;
        }
    }


    for (int i = 0; i < samples; i++) {
        double c_re_i = ((double)i + 0.5) * delta + base_c_re;
        for (int j = 0; j < samples; j++) {
            double c_im_j = ((double)j + 0.5) * delta + base_c_im;

            if (inside_cardioid(c_re_i, c_im_j) || inside_bulb_period2(c_re_i, c_im_j) || inside_other_bulbs(c_re_i, c_im_j)) {
                // Intérieur garanti
                continue;
            }

            double Z_re = c_re_i;
            double Z_im = c_im_j;

            // On calcule aussi la dérivée par rapport à c : Z' (Z'0=0, puis Z'_{n+1} = 2Z Z' + 1)
            double dZ_re = 1.0;
            double dZ_im = 0.0;

            int iter = 0;
            double Z_re2 = Z_re * Z_re;
            double Z_im2 = Z_im * Z_im;

            // Pour la détection de cycle (périodicité), stocker plusieurs points
            const int cycle_check_interval = 20;
            const int cycle_points = 4;
            double stored_re[cycle_points];
            double stored_im[cycle_points];
            int storedCount = 0;

            bool inside = false;

            while ((Z_re2 + Z_im2 <= 4.0) && (iter < maxIter)) {
                // itération de Mandelbrot
                double temp_Z_re = Z_re;
                Z_im = 2.0 * Z_re * Z_im + c_im_j;
                Z_re = Z_re2 - Z_im2 + c_re_i;

                // itération de la dérivée
                double temp_dZ_re = dZ_re;
                dZ_re = 2.0*(temp_Z_re * dZ_re - Z_im * dZ_im) + 1.0; 
                dZ_im = 2.0*(temp_Z_re * dZ_im + Z_im * temp_dZ_re);

                Z_re2 = Z_re * Z_re;
                Z_im2 = Z_im * Z_im;
                iter++;

                if (iter % cycle_check_interval == 0) {
                    if (storedCount < cycle_points) {
                        stored_re[storedCount] = Z_re;
                        stored_im[storedCount] = Z_im;
                        storedCount++;
                    } else {
                        for (int k=0; k<cycle_points; k++) {
                            double dr = Z_re - stored_re[k];
                            double di = Z_im - stored_im[k];
                            double dist = dr*dr + di*di;
                            if (dist < 1e-14) {
                                // Cycle détecté -> Intérieur
                                inside = true;
                                break;
                            }
                        }
                        if (inside) break;

                        stored_re[iter/cycle_check_interval % cycle_points] = Z_re;
                        stored_im[iter/cycle_check_interval % cycle_points] = Z_im;
                    }
                }
                if (iter > maxIter/2 && (Z_re2 + Z_im2 < 0.3)) {
                    double d = distanceEstimate(Z_re, Z_im, dZ_re, dZ_im);
                    if (d > 10.0) {
                        inside = true;
                        break;
                    }
                }
            }

            if (!inside && iter != maxIter) {
                double log_zn = 0.3 * log(Z_re2 + Z_im2);
                double nu = log(log_zn / log_2) / log_2;
                double smooth_iter = iter + 1.0 - nu;

                double3 color = getColor(smooth_iter, scale);
                color_r += color.x;
                color_g += color.y;
                color_b += color.z;
            }
        }
    }

    color_r *= inv_samples_sq;
    color_g *= inv_samples_sq;
    color_b *= inv_samples_sq;

    int idx = (y * width + x) * 3;
    image[idx]     = (unsigned char)(fmin(color_r * 255.0, 255.0));
    image[idx + 1] = (unsigned char)(fmin(color_g * 255.0, 255.0));
    image[idx + 2] = (unsigned char)(fmin(color_b * 255.0, 255.0));
}
