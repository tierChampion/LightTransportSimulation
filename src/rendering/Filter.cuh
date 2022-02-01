#ifndef __filter_cuh__
#define __filter_cuh__

#include "../geometry/Geometry.cuh"

namespace lts {

	class Filter {

	public:

		const Vector2f radius, invRadius;

		__host__ Filter(const Vector2f& r) : radius(r), invRadius(Vector2f(1 / r.x, 1 / r.y)) {}

		__host__ virtual ~Filter() {}

		__host__ virtual float evaluate(const Point2f& p) const = 0;

		__host__ float* getFilterTable(int d) {

			float* table = new float[d * d];

			for (int y = 0; y < d; y++) {
				for (int x = 0; x < d; x++) {

					Point2f p;

					p.x = (x + 0.5f) * radius.x / d;
					p.y = (y + 0.5f) * radius.y / d;

					table[y * d + x] = evaluate(p);
				}
			}

			return table;
		}
	};

	class BoxFilter : public Filter {

	public:

		__host__ BoxFilter(const Vector2f& r) : Filter(r) {}
		__host__ float evaluate(const Point2f& p) const override { return 1.0f; }

	};

	class TriangleFilter : public Filter {

	public:

		__host__ TriangleFilter(const Vector2f& r) : Filter(r) {}
		__host__ float evaluate(const Point2f& p) const override {
			return fmaxf(0.0f, radius.x - fabsf(p.x)) *
				fmaxf(0.0f, radius.y - fabsf(p.y));
		}
	};

	class GaussianFilter : public Filter {

		const float alpha;
		const float expX, expY;

		__host__ float gaussian(float d, float expv) const {
			return fmaxf(0, expf(-alpha * d * d) - expv);
		}

	public:

		__host__ GaussianFilter(const Vector2f& r, float a) : Filter(r),
			alpha(a), expX(expf(-alpha * radius.x * radius.x)), expY(expf(-alpha * radius.y * radius.y))
		{}
		__host__ float evaluate(const Point2f& p) const override {
			return gaussian(p.x, expX) * gaussian(p.y, expY);
		}
	};

	class MitchellFilter : public Filter {

		// Have B + 2C = 1 for better results
		const float B, C;

		__host__ float mitchell1D(float x) const {
			x = fabsf(2 * x);
			if (x > 1) {
				return ((-B - 6 * C) * (x * x * x) +
					(6 * B + 30 * C) * (x * x) +
					(-12 * B - 48 * C) * x +
					(8 * B + 24 * C)) *
					(1.f / 6.f);
			}
			else {
				return ((12 - 9 * B - 6 * C) * (x * x * x) +
					(-18 + 12 * B + 6 * C) * (x * x) +
					(6 - 2 * B)) *
					(1.f / 6.f);
			}
		}

	public:

		__host__ MitchellFilter(const Vector2f& r, float b, float c) : Filter(r),
			B(b), C(c)
		{}

		__host__ float evaluate(const Point2f& p) const override {
			return mitchell1D(p.x * invRadius.x) * mitchell1D(p.y * invRadius.y);
		}
	};

	class LanczosSincFilter : public Filter {

		float tau;

		__host__ float sinc(float x) const {
			x = fabsf(x);
			if (x < SHADOW_EPSILON) return 1;
			return sinf(M_PI * x) / (M_PI * x);
		}

		__host__ float windowedSinc(float x, float radius) const {
			x = fabsf(x);
			if (x > radius) return 0;
			float lanczos = sinc(x / tau);
			return sinc(x) * lanczos;
		}

	public:

		__host__ LanczosSincFilter(Vector2f& r, float t) : Filter(r), tau(t) {}

		__host__ float evaluate(const Point2f& p) const override {
			return windowedSinc(p.x, radius.x) * windowedSinc(p.y, radius.y);
		}
	};
}

#endif