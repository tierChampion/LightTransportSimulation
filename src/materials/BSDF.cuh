#ifndef __bsdf_cuh__
#define __bsdf_cuh__

#include "models/BxDF.cuh"
#include "../geometry/Interaction.cuh"

namespace lts {

	// use arena memory allocation 

	class BSDF {

		const Normal3f ns, ng;
		const Vector3f ss, ts;
		int nBxDFs;
		static constexpr int MaxBxDFs = 2;
		BxDF* bxdfs[MaxBxDFs];
		//friend mixbxdf

	public:

		const float eta;

		__device__ BSDF(const SurfaceInteraction& si, float eta = 1) : eta(eta), ns(si.shading.n), ng(si.it.n),
			ss(normalize(si.shading.dpdu)), ts(cross(Vector3f(ns), ss)) {
			nBxDFs = 0;
		}

		__device__ void clean() {
			for (int i = 0; i < nBxDFs; i++) {
				bxdfs[i]->clean();
				free(bxdfs[i]);
			}
		}


		__device__ int numComponents(BxDFType flags = BSDF_ALL) const {
			int num = 0;
			for (int i = 0; i < nBxDFs; i++) {
				if (bxdfs[i]->matchesFlag(flags)) num++;
			}
			return num;
		}

		__device__ void add(BxDF* b) {
			assert(nBxDFs < MaxBxDFs);
			bxdfs[nBxDFs++] = b;
		}

		__device__ Vector3f worldToLocal(const Vector3f& v) const {
			return Vector3f(dot(v, ss), dot(v, ts), dot(v, ns));
		}

		__device__ Vector3f localToWorld(const Vector3f& v) const {
			return Vector3f(ss.x * v.x + ts.x * v.y + ns.x * v.z,
				ss.y * v.x + ts.y * v.y + ns.y * v.z,
				ss.z * v.x + ts.z * v.y + ns.z * v.z);
		}

		__device__ Spectrum f(const Vector3f& woW, const Vector3f& wiW, BxDFType flags = BSDF_ALL) const {

			Vector3f wi = worldToLocal(wiW), wo = worldToLocal(woW);
			bool reflect = dot(wiW, ng) * dot(woW, ng) > 0;
			Spectrum f(0.0f);
			for (int i = 0; i < nBxDFs; i++) {
				if (bxdfs[i]->matchesFlag(flags) &&
					((reflect && (bxdfs[i]->type & BSDF_REFLECTION)) ||
						(!reflect && (bxdfs[i]->type & BSDF_TRANSMISSION))))
					f += bxdfs[i]->f(wo, wi);
			}
			return f;
		}

		__device__ Spectrum sampleF(const Vector3f& woWorld, Vector3f* wiWorld,
			const Point2f& sample, float* pdf, BxDFType type, BxDFType* sampleType) const {

			int matchingComps = numComponents(type);
			if (matchingComps == 0) {
				*pdf = 0;
				return Spectrum(0.0f);
			}
			int comp = fminf((int)floorf(sample.x * matchingComps),
				matchingComps - 1);

			BxDF* bxdf = nullptr;
			int count = comp;
			for (int i = 0; i < nBxDFs; i++) {
				if (bxdfs[i]->matchesFlag(type) && count-- == 0) {
					bxdf = bxdfs[i];
					break;
				}
			}

			Point2f remappedSample(sample.x * matchingComps - comp, sample.y);
			// sample bxdf
			Vector3f wi, wo = worldToLocal(woWorld);
			*pdf = 0;
			if (sampleType) *sampleType = bxdf->type;
			Spectrum f = bxdf->sampleF(wo, &wi, remappedSample, pdf, sampleType);

			if (*pdf == 0)
				return 0.0;
			*wiWorld = localToWorld(wi);

			if (!(bxdf->type & BSDF_SPECULAR) && matchingComps > 1) {
				for (int i = 0; i < nBxDFs; i++) {
					if (bxdfs[i] != bxdf && bxdfs[i]->matchesFlag(type)) {
						*pdf += bxdfs[i]->Pdf(wo, wi);
					}
				}
			}
			if (matchingComps > 1) *pdf /= matchingComps;

			if (!(bxdf->type & BSDF_SPECULAR) && matchingComps > 1) {
				bool reflect = dot(*wiWorld, ng) * dot(woWorld, ng) > 0;
				f = 0.0;
				for (int i = 0; i < nBxDFs; i++) {
					if (bxdfs[i]->matchesFlag(type) &&
						((reflect && (bxdfs[i]->type & BSDF_REFLECTION)) ||
							(!reflect && (bxdfs[i]->type & BSDF_TRANSMISSION))))
						f += bxdfs[i]->f(wo, wi);
				}
			}

			return f;

		}

		__device__ float Pdf(const Vector3f& woWorld, const Vector3f& wiWorld, BxDFType flags = BSDF_ALL) const {
			if (nBxDFs == 0) return 0.0f;
			Vector3f wo = worldToLocal(woWorld), wi = worldToLocal(wiWorld);
			if (wo.z == 0) return 0.0f;
			float pdf = 0.0f;
			int matchingComps = 0;
			for (int i = 0; i < nBxDFs; i++) {
				if (bxdfs[i]->matchesFlag(flags)) {
					matchingComps++;
					//	if (bxdfs[i]) printf("No bxdfs!\n");
					pdf = pdf + bxdfs[i]->Pdf(wo, wi);
				}
			}
			float v = matchingComps > 0 ? pdf / matchingComps : 0.0f;
			return v;
		}
	};
}

#endif