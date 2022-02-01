#ifndef __transform_cuh__
#define __transform_cuh__

#include "Geometry.cuh"
#include "Interaction.cuh"

namespace lts {

	struct Quaternion {

		float q[4];

		__host__ __device__ Quaternion() { q[0] = 1.0f; q[1] = q[2] = q[3] = 0.0f; }
		__host__ __device__ Quaternion(float w, float x, float y, float z) { q[0] = w; q[1] = x; q[2] = y; q[3] = z; }

		__host__ __device__ bool operator==(const Quaternion& q) const {
			return (this->q[0] == q.q[0] &&
				this->q[1] == q.q[1] &&
				this->q[2] == q.q[2] &&
				this->q[3] == q.q[3]);
		}

		__host__ __device__ bool operator!=(const Quaternion& q) const {
			return !(*this == q);
		}

		__host__ __device__ Vector3f getAxisPart() const {
			return Vector3f(q[1], q[2], q[3]);
		}

		// http://www.songho.ca/opengl/gl_quaternion.html
		__host__ __device__ void toMatrix(float mat[16]) const {
			mat[0] = 1 - 2 * (q[2] * q[2] - q[3] * q[3]);
			mat[1] = 2 * (q[1] * q[2] - q[0] * q[3]);
			mat[2] = 2 * (q[1] * q[3] + q[0] * q[2]);
			mat[3] = 0.0f;
			mat[4] = 2 * (q[1] * q[2] + q[0] * q[3]);
			mat[5] = 1 - 2 * (q[1] * q[1] - q[3] * q[3]);
			mat[6] = 2 * (q[2] * q[3] - q[0] * q[1]);
			mat[7] = 0.0f;
			mat[8] = 2 * (q[1] * q[3] - q[0] * q[2]);
			mat[9] = 2 * (q[2] * q[3] + q[0] * q[1]);
			mat[10] = 1 - 2 * (q[1] * q[1] - q[2] * q[2]);
			mat[11] = 0.0f;
			mat[12] = 0.0f;
			mat[13] = 0.0f;
			mat[14] = 0.0f;
			mat[15] = 1.0f;
		}

		__host__ __device__ float magnitudeSquared() const {
			return q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
		}
	};

	__host__ __device__ inline Quaternion quaternionMult(const Quaternion& q1, const Quaternion& q2) {

		float w = q1.q[0] * q2.q[0] - q1.q[1] * q2.q[1] - q1.q[2] * q2.q[2] - q1.q[3] * q2.q[3];
		float x = q1.q[0] * q2.q[1] + q1.q[1] * q2.q[0] + q1.q[2] * q2.q[3] - q1.q[3] * q2.q[2];
		float y = q1.q[0] * q2.q[2] - q1.q[1] * q2.q[3] + q1.q[2] * q2.q[0] + q1.q[3] * q2.q[1];
		float z = q1.q[0] * q2.q[3] + q1.q[1] * q2.q[2] - q1.q[2] * q2.q[1] + q1.q[3] * q2.q[0];
		return Quaternion(w, x, y, z);
	}

	template <typename T>
	__device__ inline Vector3<T> absQuaternionMultError(const Quaternion& qToTest, const Vector3<T>& v) {

		T x = v.x, y = v.y, z = v.z;

		float m[16];

		qToTest.toMatrix(m);

		T xAbsSum = (fabsf(m[0] * x) + fabsf(m[1] * y) +
			fabsf(m[2] * z) + fabsf(m[3]));
		T yAbsSum = (fabsf(m[4] * x) + fabsf(m[5] * y) +
			fabsf(m[6] * z) + fabsf(m[7]));
		T zAbsSum = (fabsf(m[8] * x) + fabsf(m[9] * y) +
			fabsf(m[10] * z) + fabsf(m[11]));

		return Vector3<T>(xAbsSum, yAbsSum, zAbsSum) * gamma(3);
	}

	template <typename T>
	__device__ inline Vector3<T> absQuaternionMultError(const Quaternion& qToTest, const Vector3<T>& v, const Vector3<T>& vError) {

		T x = v.x, y = v.y, z = v.z;

		float m[16];

		qToTest.toMatrix(m);

		T xAbsSum = (gamma(3) + (T)1) * (fabsf(m[0] * vError.x) + fabsf(m[1] * vError.y) + fabsf(m[2] * vError.z)) +
			gamma(3) * (fabsf(m[0] * x) + fabsf(m[1] * y) + fabsf(m[2] * z));

		T yAbsSum = (gamma(3) + (T)1) * (fabsf(m[4] * vError.x) + fabsf(m[5] * vError.y) + fabsf(m[6] * vError.z)) +
			gamma(3) * (fabsf(m[4] * x) + fabsf(m[5] * y) + fabsf(m[6] * z));

		T zAbsSum = (gamma(3) + (T)1) * (fabsf(m[8] * vError.x) + fabsf(m[9] * vError.y) + fabsf(m[10] * vError.z)) +
			gamma(3) * (fabsf(m[8] * x) + fabsf(m[9] * y) + fabsf(m[10] * z));

		return Vector3<T>(xAbsSum, yAbsSum, zAbsSum) * gamma(3);
	}

	__host__ __device__ inline Quaternion normalize(Quaternion& q) {
		float invSqrt = quakeIIIFastInvSqrt(q.magnitudeSquared());
		q.q[0] *= invSqrt;
		q.q[1] *= invSqrt;
		q.q[2] *= invSqrt;
		q.q[3] *= invSqrt;
		return q;
	}

	__host__ __device__ inline Quaternion conjugate(const Quaternion q) {
		return Quaternion(q.q[0], -q.q[1], -q.q[2], -q.q[3]);
	}

	__host__ __device__ inline Quaternion axisAngleToQuaternion(const Vector3f& axis, float theta) {
		float w = cosf(theta / 2);
		float x = axis.x * sinf(theta / 2);
		float y = axis.y * sinf(theta / 2);
		float z = axis.z * sinf(theta / 2);
		return Quaternion(w, x, y, z);
	}

	__host__ __device__ inline Quaternion quaternionToAxisAngle(const Quaternion& q) {
		float w = acosf(q.q[0]) * 2;
		return Quaternion(w, q.q[1], q.q[2], q.q[3]);
	}

	class Transform {

		Vector3f translation;
		Quaternion rotation;

	public:

		__host__ __device__ Transform() {}

		__host__ __device__ Transform(const Vector3f& t) {
			translation = t;
			rotation = Quaternion();
		}

		__host__ __device__ Transform(const Vector3f& axis, float theta) {
			translation = Vector3f();
			rotation = axisAngleToQuaternion(axis, fmodf(theta, 2 * M_PI));
		}

		__host__ __device__ Transform(const Vector3f& t, const Vector3f axis, float theta) {
			translation = t;
			rotation = axisAngleToQuaternion(axis, fmodf(theta, 2 * M_PI));
		}

		__host__ __device__ Transform(const Vector3f& t, const Quaternion& q) {
			translation = t;
			rotation = q;
		}
		// not sure if works...
		__host__ __device__ Transform getInverse() const {
			return Transform(-translation, conjugate(rotation));
		}

		__host__ __device__ Transform operator*(const Transform& t) const {
			return Transform(translation + t.translation, quaternionMult(rotation, t.rotation));
		}

		// ABSOLUTE TRANSFORMATIONS
		template <typename T>
		__host__ __device__ inline Vector3<T> operator()(const Vector3<T>& v) const;
		template <typename T>
		__host__ __device__ inline Point3<T> operator()(const Point3<T>& p) const;
		template <typename T>
		__host__ __device__ inline Normal3<T> operator()(const Normal3<T>& n) const;
		template <typename T>
		__host__ __device__ inline Bounds3<T> operator()(const Bounds3<T>& b) const;
		__device__ inline Ray operator()(const Ray& r) const;
		__device__ inline Interaction operator()(const Interaction& it) const;
		__device__ inline SurfaceInteraction operator()(const SurfaceInteraction& si) const;

		// ERROR TRANSFORMATIONS
		template <typename T>
		__host__ __device__ inline Point3<T> Transform::operator()(const Point3<T>& p, Vector3<T>* absError) const;
		template <typename T>
		__host__ __device__ inline Vector3<T> Transform::operator()(const Vector3<T>& v, Vector3<T>* absError) const;
		template <typename T>
		__host__ __device__ inline Point3<T> Transform::operator()
			(const Point3<T>& p, const Vector3<T>& pError, Vector3<T>* absError) const;
		template <typename T>
		__host__ __device__ inline Vector3<T> Transform::operator()
			(const Vector3<T>& v, const Vector3<T>& vError, Vector3<T>* absError) const;

		__device__ inline Ray Transform::operator()(const Ray& r, Vector3f* oError, Vector3f* dError) const;

		__device__ inline Ray Transform::operator()(const Ray& r,
			Vector3f& oErrorIn, Vector3f& dErrorIn, Vector3f* oErrorOut, Vector3f* dErrorOut) const;
	};

	template <typename T>
	__host__ __device__ inline Vector3<T> Transform::operator()(const Vector3<T>& v) const {
		Quaternion q = Quaternion(0.0f, v.x, v.y, v.z);
		Quaternion transformedQ = quaternionMult(quaternionMult(rotation, q), conjugate(rotation));
		return quaternionToAxisAngle(transformedQ).getAxisPart() + translation;
	}

	template <typename T>
	__host__ __device__ inline Point3<T> Transform::operator()(const Point3<T>& p) const {
		Vector3f v = (Vector3f)p;
		return (Point3<T>)((*this)(v));
	}

	template <typename T>
	__host__ __device__ inline Normal3<T> Transform::operator()(const Normal3<T>& n) const {
		Quaternion q = Quaternion(0.0f, n.x, n.y, n.z);
		Quaternion transformedQ = quaternionMult(quaternionMult(conjugate(rotation), q), rotation);

		return (Normal3<T>)normalize(quaternionToAxisAngle(transformedQ).getAxisPart());
	}

	template <typename T>
	__host__ __device__ inline Bounds3<T> Transform::operator()(const Bounds3<T>& b) const {
		const Transform& M = *this;
		Bounds3f ret(M(Point3f(b.pMin.x, b.pMin.y, b.pMin.z)));
		for (int c = 1; c < 8; c++) {
			ret = Union(ret, M(b.corner(c)));
		}

		return ret;
	}

	__device__ inline Ray Transform::operator()(const Ray& r) const {

		Vector3f oError;
		Point3f o = (*this)(r.o, &oError);
		Vector3f d = (*this)(r.d);
		float lengthSquared = d.lengthSquared();
		float tMax = r.tMax;
		if (lengthSquared > 0) {
			float dt = dot(abs(d), oError) / lengthSquared;
			o += d * dt;
			tMax -= dt;
		}
		return Ray(o, d, tMax);
	}

	template <typename T>
	__device__ inline Point3<T> Transform::operator()(const Point3<T>& p, Vector3<T>* absError) const {

		Vector3f v = (Vector3f)p;
		*absError = absQuaternionMultError(rotation, v);

		return (*this)(p);
	}

	template <typename T>
	__device__ inline Vector3<T> Transform::operator()(const Vector3<T>& v, Vector3<T>* absError) const {

		*absError = absQuaternionMultError(rotation, v);
		return (*this)(v);
	}

	template <typename T>
	__device__ inline Point3<T> Transform::operator()(const Point3<T>& p, const Vector3<T>& pError,
		Vector3<T>* absError) const {

		Vector3f v = (Vector3f)p;
		*absError = absQuaternionMultError(rotation, v, pError);

		return (*this)(p);
	}


	template <typename T>
	__device__ inline Vector3<T> Transform::operator()(const Vector3<T>& v, const Vector3<T>& vError,
		Vector3<T>* absError) const {

		*absError = absQuaternionMultError(rotation, v, vError);

		return (*this)(v);
	}

	__device__ inline Ray Transform::operator()(const Ray& r, Vector3f* oError, Vector3f* dError) const {
		Point3f o = (*this)(r.o, oError);
		Vector3f d = (*this)(r.d, dError);
		float lengthSquared = d.lengthSquared();
		float tMax = r.tMax;
		if (lengthSquared > 0) {
			float dt = dot(abs(d), *oError) / lengthSquared;
			o += d * dt;
			tMax -= dt;
		}
		return Ray(o, d, tMax);
	}

	__device__ inline Ray Transform::operator()(const Ray& r,
		Vector3f& oErrorIn, Vector3f& dErrorIn,
		Vector3f* oErrorOut, Vector3f* dErrorOut) const {

		Point3f o = (*this)(r.o, oErrorIn, oErrorOut);
		Vector3f d = (*this)(r.d, dErrorIn, dErrorOut);
		float lengthSquared = d.lengthSquared();
		float tMax = r.tMax;
		if (lengthSquared > 0) {
			float dt = dot(abs(d), *oErrorOut) / lengthSquared;
			o += d * dt;
			tMax -= dt;
		}
		return Ray(o, d, tMax);
	}

	__device__ inline Interaction Transform::operator()(const Interaction& it) const {

		Interaction ret;

		ret.p = (*this)(it.p, it.pError, &ret.pError);

		const Transform& t = *this;
		ret.n = normalize(t(it.n));
		ret.wo = t(it.wo);

		return ret;
	}

	__device__ inline SurfaceInteraction Transform::operator()(const SurfaceInteraction& si) const {

		SurfaceInteraction ret;

		ret.it.p = (*this)(si.it.p, si.it.pError, &ret.it.pError);

		const Transform& t = *this;
		ret.it.n = normalize(t(si.it.n));
		ret.it.wo = t(si.it.wo);
		ret.uv = si.uv;

		ret.dpdu = t(si.dpdu);
		ret.dpdv = t(si.dpdv);
		ret.dndu = t(si.dndu);
		ret.dndv = t(si.dndv);
		ret.shading.n = normalize(t(si.shading.n));
		ret.shading.dpdu = t(si.shading.dpdu);
		ret.shading.dpdv = t(si.shading.dpdv);
		ret.shading.dndu = t(si.shading.dndu);
		ret.shading.dndv = t(si.shading.dndv);
		ret.dudx = si.dudx;
		ret.dvdx = si.dvdx;
		ret.dudy = si.dudy;
		ret.dvdy = si.dvdy;
		ret.dpdx = t(si.dpdx);
		ret.dpdy = t(si.dpdy);
		ret.tri = si.tri;
		ret.primitive = si.primitive;
		ret.bsdf = si.bsdf;
		ret.shading.n = faceTowards(ret.shading.n, ret.it.n);

		return ret;
	}
}

#endif