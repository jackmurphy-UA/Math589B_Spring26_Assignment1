#include <cmath>
#include <algorithm>

extern "C" {

// Bump when you change the exported function signatures.
int rod_api_version() { return 2; }

// Helper clamp
static inline double clamp01(double s) {
    return (s < 0.0) ? 0.0 : (s > 1.0 ? 1.0 : s);
}

// Closest points between segments p1->q1 and p2->q2.
// Returns parameters s,t in [0,1] minimizing || (p1 + s d1) - (p2 + t d2) ||.
// Implementation adapted from standard segment-segment distance derivations (Ericson-style).
static inline void closest_segment_params(
    const double p1[3], const double q1[3],
    const double p2[3], const double q2[3],
    double &s, double &t
) {
    const double d1[3] = { q1[0]-p1[0], q1[1]-p1[1], q1[2]-p1[2] };
    const double d2[3] = { q2[0]-p2[0], q2[1]-p2[1], q2[2]-p2[2] };
    const double r[3]  = { p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2] };

    const double a = d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2]; // ||d1||^2
    const double e = d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2]; // ||d2||^2
    const double b = d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2];
    const double c = d1[0]*r[0]  + d1[1]*r[1]  + d1[2]*r[2];
    const double f = d2[0]*r[0]  + d2[1]*r[1]  + d2[2]*r[2];

    const double EPS = 1e-14;

    // Handle degenerate segments
    if (a <= EPS && e <= EPS) {
        // both are points
        s = 0.0; t = 0.0;
        return;
    }
    if (a <= EPS) {
        // first is a point
        s = 0.0;
        t = clamp01(f / e);
        return;
    }
    if (e <= EPS) {
        // second is a point
        t = 0.0;
        s = clamp01(-c / a);
        return;
    }

    // General case
    const double denom = a*e - b*b;

    if (denom > EPS) {
        s = clamp01((b*f - c*e) / denom);
    } else {
        // nearly parallel
        s = 0.0;
    }

    // Solve for t given s
    t = (b*s + f) / e;

    // Clamp t, and recompute s if needed (standard segment-segment clamping logic)
    if (t < 0.0) {
        t = 0.0;
        s = clamp01(-c / a);
    } else if (t > 1.0) {
        t = 1.0;
        s = clamp01((b - c) / a);
    }
}

// Exported API
void rod_energy_grad(
    int N,
    const double* x,
    double kb,
    double ks,
    double l0,
    double kc,     // confinement strength
    double eps,    // WCA epsilon
    double sigma,  // WCA sigma
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

    // ---- Segment–segment WCA self-avoidance ----
    // Segments: (i,i+1), periodic. Exclude adjacent segments (including wrap).
    // WCA cutoff: rc = 2^(1/6) * sigma
    const double rc = std::pow(2.0, 1.0/6.0) * sigma;
    const double rc2 = rc * rc;
    const double tiny = 1e-12;

    for (int i = 0; i < N; ++i) {
        // segment i: Pi -> Qi
        double Pi[3] = { get(i,0), get(i,1), get(i,2) };
        double Qi[3] = { get(i+1,0), get(i+1,1), get(i+1,2) };

        for (int j = i+1; j < N; ++j) {
            // Exclusions: skip same/adjacent segments, including wrap neighbors.
            // Adjacent if |i-j| <= 1 or |i-j| >= N-1
            int dj = j - i;
            if (dj <= 1) continue;
            if (dj >= N-1) continue;

            double Pj[3] = { get(j,0), get(j,1), get(j,2) };
            double Qj[3] = { get(j+1,0), get(j+1,1), get(j+1,2) };

            double u, v;
            closest_segment_params(Pi, Qi, Pj, Qj, u, v);

            // Closest points
            const double Di[3] = { Qi[0]-Pi[0], Qi[1]-Pi[1], Qi[2]-Pi[2] };
            const double Dj[3] = { Qj[0]-Pj[0], Qj[1]-Pj[1], Qj[2]-Pj[2] };

            double Ci[3] = { Pi[0] + u*Di[0], Pi[1] + u*Di[1], Pi[2] + u*Di[2] };
            double Cj[3] = { Pj[0] + v*Dj[0], Pj[1] + v*Dj[1], Pj[2] + v*Dj[2] };

            double rx = Ci[0] - Cj[0];
            double ry = Ci[1] - Cj[1];
            double rz = Ci[2] - Cj[2];

            double d2 = rx*rx + ry*ry + rz*rz;
            if (d2 >= rc2) continue;

            double d = std::sqrt(std::max(d2, tiny));

            // WCA energy: 4 eps [ (sigma/d)^12 - (sigma/d)^6 ] + eps
            const double invd = 1.0 / d;
            const double sr = sigma * invd;
            const double sr2 = sr * sr;
            const double sr6 = sr2 * sr2 * sr2;
            const double sr12 = sr6 * sr6;
            const double U = 4.0 * eps * (sr12 - sr6) + eps;
            E += U;

            // dU/dd = 24 eps * (sr6 - 2 sr12) / d
            // Gradient wrt coordinates: dU/dx = (dU/dd) * (r/d) * weight
            // Combine: factor = 24 eps (sr6 - 2 sr12) / d^2
            const double factor = 24.0 * eps * (sr6 - 2.0*sr12) * (invd * invd);

            // r vector
            const double gx = factor * rx;
            const double gy = factor * ry;
            const double gz = factor * rz;

            // Distribute to endpoints using linear interpolation weights
            const double wi0 = (1.0 - u);
            const double wi1 = u;
            const double wj0 = (1.0 - v);
            const double wj1 = v;

            // Segment i endpoints: + (wi0, wi1) * g
            addg(i,   0, wi0 * gx);
            addg(i,   1, wi0 * gy);
            addg(i,   2, wi0 * gz);
            addg(i+1, 0, wi1 * gx);
            addg(i+1, 1, wi1 * gy);
            addg(i+1, 2, wi1 * gz);

            // Segment j endpoints: - (wj0, wj1) * g
            addg(j,   0, -wj0 * gx);
            addg(j,   1, -wj0 * gy);
            addg(j,   2, -wj0 * gz);
            addg(j+1, 0, -wj1 * gx);
            addg(j+1, 1, -wj1 * gy);
            addg(j+1, 2, -wj1 * gz);
        }
    }

    *energy_out = E;
}

} // extern "C"
