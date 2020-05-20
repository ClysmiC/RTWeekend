#pragma once

#include "als/als.h"
#include <ostream>

struct IHittable;
struct IMaterial;
struct BvhNode;

enum KZERO
{
    kZero
};

struct Vector3
{
    union
    {
        struct
        {
            float x;
            float y;
            float z;
        };

        float elements[3];
    };

	Vector3() : x(0), y(0), z(0) { }
    Vector3(KZERO) : x(0), y(0), z(0) { }
	Vector3(float x, float y, float z) : x(x), y(y), z(z) { }

	float length() const;
	float lengthSquared() const;
	void normalizeInPlace();
};

float dot(const Vector3 & lhs, const Vector3 & rhs);
Vector3 cross(const Vector3 & lhs, const Vector3 & rhs);
Vector3 normalize(const Vector3 & v);
Vector3 operator+(const Vector3 & v);
Vector3 operator-(const Vector3 & v);
void operator+=(Vector3 & vLeft, const Vector3 & vRight);
void operator-=(Vector3 & vLeft, const Vector3 & vRight);
void operator*=(Vector3 & v, float scale);
void operator/=(Vector3 & v, float scale);
Vector3 operator+ (const Vector3 & vLeft, const Vector3 & vRight);
Vector3 operator- (const Vector3 & vLeft, const Vector3 & vRight);
Vector3 hadamard(const Vector3 & vLeft, const Vector3 & vRight);
Vector3 operator* (const Vector3 & v, float f);
Vector3 operator* (float f, const Vector3 & v);
Vector3 operator/ (const Vector3 & v, float f);
std::ostream & operator<<(std::ostream & outputStream, const Vector3 & vector);

Vector3 randomVectorInsideUnitSphere();
Vector3 randomVectorInsideUnitDisk();

Vector3 reflect(const Vector3 & v, const Vector3 & n);
bool refractWithSchlickProbability(const Vector3 & v, const Vector3 & n, float refractionIndexFrom, float refractionIndexTo, Vector3 * vecOut);

float rand0Incl1Excl();

template<typename T>
inline float lerp(T val0, T val1, float t)
{
    return val0 + t * (val1 - val0);
}

struct Ray
{
	Vector3 p0;
	Vector3 dir;
    float time;     // Scene time. Not to be confused with parametric "t"

	Ray() : p0(), dir(), time() { }
	Ray(Vector3 p0, Vector3 dir, float time) : p0(p0), dir(normalize(dir)), time(time) { }

	Vector3 pointAtT(float t) const;
	// Vector3 color(IHittable ** aHittable, int cHittable, int rayDepth) const;
    Vector3 color(BvhNode * bvhNode, int rayDepth) const;
};

struct Aabb
{
    Vector3 min;
    Vector3 max;

    Aabb() : min(), max() { };
    Aabb(Vector3 p0, Vector3 p1);
    Aabb(Aabb aabb0, Aabb aabb1);

    bool testHit(const Ray & ray, float tMin, float tMax) const;
};

struct Camera
{
    Vector3 pos;
    Vector3 forward;
    Vector3 right;
    Vector3 up;

    float fovDeg;
    float aspectRatio;
    float lensRadius;

    // Motion blur

    float time0;
    float time1;

    // Cached view plane information

    Vector3 botLeftViewPosCached;
    Vector3 topRightViewPosCached;
    float wCached;
    float hCached;

    
    Camera(
        Vector3 pos,
        Vector3 lookat,
        float fovDeg,
        float aspectRatio,
        float lensRadius,
        float time0,
        float time1);

    Ray rayAt(float s, float t);
};

struct HitRecord
{
	float t;
	Vector3 normal;
    const IHittable * hittable;

	HitRecord() : t(0), normal(), hittable(nullptr) {}
};

struct IHittable
{
    IMaterial * material;

    IHittable(IMaterial * material) : material(material) { }
    
	virtual bool testHit(const Ray & ray, float tMin, float tMax, HitRecord * hitOut) const = 0;
    virtual bool tryComputeBoundingBox(float t0, float t1, Aabb * aabbOut) const = 0;
};


struct Sphere : public IHittable
{
	Vector3 p0Center;
    Vector3 velocity;

	float radius;

    Sphere(Vector3 p0Center, float radius, IMaterial * material, Vector3 velocity)
        : IHittable(material)
        , p0Center(p0Center)
        , radius(radius)
        , velocity(velocity)
    { }

	bool testHit(const Ray & ray, float tMin, float tMax, HitRecord * hitOut) const override;
    bool tryComputeBoundingBox(float t0, float t1, Aabb * aabbOut) const override;
    Vector3 posAtTime(float time) const;
};

struct BvhNode : public IHittable
{
    IHittable * left;
    IHittable * right;
    Aabb aabb;

    BvhNode(IHittable * left, IHittable * right, float time0, float time1);

    bool testHit(const Ray & ray, float tMin, float tMax, HitRecord * hitOut) const override;
    bool tryComputeBoundingBox(float t0, float t1, Aabb * aabbOut) const override;
};

BvhNode * buildBvh(IHittable ** aHittable, int cHittable, float time0, float time1);

struct IMaterial
{
    virtual bool scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const = 0;
};

struct LambertianMaterial : public IMaterial
{
    Vector3 albedo;
    
    LambertianMaterial(const Vector3 & albedo) : albedo(albedo) { }

    bool scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const override;
};

struct MetalMaterial : public IMaterial
{
    Vector3 albedo;
	float fuzziness;
    
    MetalMaterial(const Vector3 & albedo, float fuzziness) : albedo(albedo), fuzziness(fuzziness) { }

    bool scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const override;
};

struct DielectricMaterial : public IMaterial
{
	float refractiveIndex;

	DielectricMaterial(float refractiveIndex) : refractiveIndex(refractiveIndex) { }

	bool scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const override;
};


