#include <cmath>
#include <algorithm>

extern "C" {

int rod_api_version() { return 2; }

// Robust closest parameters between segments p1->q1 and p2->q2 (Dan Sunday / geomalgorithms).
static inline void closest_segment_params(
    const double p1[3], const double q1[3],
    const double p2[3], const double q2[3],
    double &sc, double &tc
) {
    const double u[3] = { q1[0]-p1[0], q1[1]-p1[1], q1[2]-p1[2] };
    const double v[3] = { q2[0]-p2[0], q2[1]-p2[1], q2[2]-p2[2] };
    const double w[3] = { p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2] };

    const double a = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    const double b = u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
    const double c = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    const double d = u[0]*w[0] + u[1]*w[1] + u[2]*w[2];
    const double e = v[0]*w[0] + v[1]*w[1] + v[2]*w[2];

    const double SMALL = 1e-14;
    const double D = a*c - b*b;

    double sN, sD = D;
    double tN, tD = D;

    if (D < SMALL) {
        // almost parallel
        sN = 0.0;
        sD = 1.0;
        tN = e;
        tD = c;
    } else {
        sN = (b*e - c*d);
        tN = (a*e - b*d);

        if (sN < 0.0) {
            sN = 0.0;
            tN = e;
            tD = c;
        } else if (sN > sD) {
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.0) {
        tN = 0.0;
        if (-d < 0.0) {
            sN = 0.0;
        } else if (-d > a) {
            sN = sD;
        } else {
            sN = -d;
            sD = a;
        }
    } else if (tN > tD) {
        tN = tD;
        if ((-d + b) < 0.0) {
            sN = 0.0;
        } else if ((-d + b) > a) {
            sN = sD;
        } else {
            sN = (-d + b);
            sD = a;
        }
    }

    sc = (std::abs(sN) < SMALL ? 0.0 : sN / sD);
    tc = (std::abs(tN) < SMALL ? 0.0 : tN / tD);

    // safety clamp
    if (sc < 0.0) sc = 0.0;
    if (sc > 1.0) sc = 1.0;
    if (tc < 0.0) tc = 0.0;
    if (tc > 1.0) tc = 1.0;
}

// Exported API v2
void rod_energy_grad(
    int N,
    const double* x,
    double kb,
    double ks,
    double l0,
    double kc,
    double eps,
    double sigma,
    double* energy_out,
    double* grad_out
) {
    const int M = 3*N;
    for (int i = 0; i < M; ++i) grad_out[i] = 0.0;
    double E = 0.0;

    auto idx = [N](int i) {
        int r = i % N;
        return (r < 0) ? (r + N) : r;
    };
    auto get = [&](int i, int d) -> double {
        return x[3*idx(i) + d];
    };
    auto addg = [&](int i, int d, double v) {
        grad_out[3*idx(i) + d] += v;
    };

    // ---- Bending: kb * ||x_{i+1} - 2 x_i + x_{i-1}||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double b = get(i+1,d) - 2.0*get(i,d) + get(i-1,d);
            E += kb * b * b;
            const double c = 2.0 * kb * b;
            addg(i-1, d, c);
            addg(i,   d, -2.0*c);
            addg(i+1, d, c);
        }
    }

    // ---- Stretching: ks * (||x_{i+1}-x_i|| - l0)^2
    for (int i = 0; i < N; ++i) {
        double dx0 = get(i+1,0) - get(i,0);
        double dx1 = get(i+1,1) - get(i,1);
        double dx2 = get(i+1,2) - get(i,2);
        double r = std::sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2);
        r = std::max(r, 1e-12);
        double diff = r - l0;
        E += ks * diff * diff;

        double coeff = 2.0 * ks * diff / r;
        addg(i+1,0,  coeff * dx0);
        addg(i+1,1,  coeff * dx1);
        addg(i+1,2,  coeff * dx2);
        addg(i,0,   -coeff * dx0);
        addg(i,1,   -coeff * dx1);
        addg(i,2,   -coeff * dx2);
    }

    // ---- Confinement: kc * sum ||x_i||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            double xi = get(i,d);
            E += kc * xi * xi;
            addg(i,d, 2.0 * kc * xi);
        }
    }

    // ---- Segment–segment WCA self-avoidance with "within two steps" exclusion ----
    // WCA cutoff rc = 2^(1/6) * sigma, energy shifted so U(rc)=0.
    const double rc = std::pow(2.0, 1.0/6.0) * sigma;
    const double rc2 = rc * rc;
    const double tiny = 1e-12;

    for (int i = 0; i < N; ++i) {
        double Pi[3] = { get(i,0), get(i,1), get(i,2) };
        double Qi[3] = { get(i+1,0), get(i+1,1), get(i+1,2) };

        // key change (friend's rule): j starts at i+3 => excludes within two steps forward
        for (int j = i + 3; j < N; ++j) {
            int dj = j - i;

            // excludes within two steps "across wrap"
            if (dj >= N - 2) continue;

            double Pj[3] = { get(j,0), get(j,1), get(j,2) };
            double Qj[3] = { get(j+1,0), get(j+1,1), get(j+1,2) };

            double u, v;
            closest_segment_params(Pi, Qi, Pj, Qj, u, v);

            const double Di[3] = { Qi[0]-Pi[0], Qi[1]-Pi[1], Qi[2]-Pi[2] };
            const double Dj[3] = { Qj[0]-Pj[0], Qj[1]-Pj[1], Qj[2]-Pj[2] };

            const double Ci[3] = { Pi[0] + u*Di[0], Pi[1] + u*Di[1], Pi[2] + u*Di[2] };
            const double Cj[3] = { Pj[0] + v*Dj[0], Pj[1] + v*Dj[1], Pj[2] + v*Dj[2] };

            const double rx = Ci[0] - Cj[0];
            const double ry = Ci[1] - Cj[1];
            const double rz = Ci[2] - Cj[2];

            const double d2 = rx*rx + ry*ry + rz*rz;
            if (d2 >= rc2) continue;

            const double d = std::sqrt(std::max(d2, tiny));
            const double invd = 1.0 / d;

            const double sr = sigma * invd;
            const double sr2 = sr * sr;
            const double sr6 = sr2 * sr2 * sr2;
            const double sr12 = sr6 * sr6;

            // shifted WCA
            const double U = 4.0 * eps * (sr12 - sr6) + eps;
            E += U;

            // dU/dd = 24 eps (sr6 - 2 sr12) / d
            // gradient factor in coordinates = dU/dd * (r/d) = 24 eps (sr6 - 2 sr12) / d^2 * r
            const double factor = 24.0 * eps * (sr6 - 2.0*sr12) * (invd * invd);

            const double gx = factor * rx;
            const double gy = factor * ry;
            const double gz = factor * rz;

            const double wi0 = (1.0 - u);
            const double wi1 = u;
            const double wj0 = (1.0 - v);
            const double wj1 = v;

            // i endpoints: +weights * g
            addg(i,   0, wi0 * gx);  addg(i,   1, wi0 * gy);  addg(i,   2, wi0 * gz);
            addg(i+1, 0, wi1 * gx);  addg(i+1, 1, wi1 * gy);  addg(i+1, 2, wi1 * gz);

            // j endpoints: -weights * g
            addg(j,   0, -wj0 * gx); addg(j,   1, -wj0 * gy); addg(j,   2, -wj0 * gz);
            addg(j+1, 0, -wj1 * gx); addg(j+1, 1, -wj1 * gy); addg(j+1, 2, -wj1 * gz);
        }
    }

    *energy_out = E;
}

} // extern "C"
