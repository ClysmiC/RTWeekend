#pragma once

#include <ostream>

struct IHitable;
struct IMaterial;

struct Vector3
{
	float x;
	float y;
	float z;

	Vector3() : x(0), y(0), z(0) { }
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
Vector3 reflect(const Vector3 & v, const Vector3 & n);

float rand0Incl1Excl();

struct Ray
{
	Vector3 p0;
	Vector3 dir;

	Ray() : p0(), dir() { }
	Ray(Vector3 p0, Vector3 dir) : p0(p0), dir(normalize(dir)) { }

	Vector3 pointAtT(float t) const;
	Vector3 color(IHitable ** aHitable, int cHitable, int rayDepth) const;
};


struct HitRecord
{
	float t;
	Vector3 normal;
    const IHitable * hitable;

	HitRecord() : t(0), normal(), hitable(nullptr) {}
};

struct IHitable
{
    IMaterial * material;

    IHitable(IMaterial * material) : material(material) { }
    
	virtual bool testHit(const Ray & ray, float tMin, float tMax, HitRecord * hitOut) const = 0;
};


struct Sphere : public IHitable
{
	Vector3 center;
	float radius;

    Sphere(Vector3 center, float radius, IMaterial * material) : IHitable(material), center(center), radius(radius) { }

	bool testHit(const Ray & ray, float tMin, float tMax, HitRecord * hitOut) const override;
};

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


