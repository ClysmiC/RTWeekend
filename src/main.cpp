#include <fstream>
#include <math.h>

#include "main.h"

using namespace std;

float Vector3::length() const
{
	return sqrt(lengthSquared());
}

float Vector3::lengthSquared() const
{
	return x * x + y * y + z * z;
}

void Vector3::normalizeInPlace()
{
	float len = length();
	x /= len;
	y /= len;
	z /= len;
}

float dot(const Vector3 & lhs, const Vector3 & rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Vector3 cross(const Vector3 & lhs, const Vector3 & rhs)
{
	return Vector3(
		lhs.y * rhs.z - lhs.z * rhs.y,
		lhs.z * rhs.x - lhs.x * rhs.z,
		lhs.x * rhs.y - lhs.y * rhs.x
	);
}

Vector3 normalize(const Vector3 & v)
{
	Vector3 result = v;
	float len = v.length();
	result.x /= len;
	result.y /= len;
	result.z /= len;

	return result;
}

Vector3 operator+(const Vector3 & v)
{
	return Vector3(v.x, v.y, v.z);
}

Vector3 operator-(const Vector3 & v)
{
	return Vector3(-v.x, -v.y, -v.z);
}

void operator+=(Vector3 & vLeft, const Vector3 & vRight)
{
	vLeft.x += vRight.x;
	vLeft.y += vRight.y;
	vLeft.z += vRight.z;
}

void operator-=(Vector3 & vLeft, const Vector3 & vRight)
{
	vLeft.x -= vRight.x;
	vLeft.y -= vRight.y;
	vLeft.z -= vRight.z;
}

void operator*=(Vector3 & v, float scale)
{
	v.x *= scale;
	v.y *= scale;
	v.z *= scale;
}

void operator/=(Vector3 & v, float scale)
{
	v.x /= scale;
	v.y /= scale;
	v.z /= scale;
}

Vector3 operator+ (const Vector3 & vLeft, const Vector3 & vRight)
{
	return Vector3(
		vLeft.x + vRight.x,
		vLeft.y + vRight.y,
		vLeft.z + vRight.z
	);
}

Vector3 operator- (const Vector3 & vLeft, const Vector3 & vRight)
{
	return Vector3(
		vLeft.x - vRight.x,
		vLeft.y - vRight.y,
		vLeft.z - vRight.z
	);
}

Vector3 hadamard(const Vector3 & vLeft, const Vector3 & vRight)
{
	return Vector3(
		vLeft.x * vRight.x,
		vLeft.y * vRight.y,
		vLeft.z * vRight.z
	);
}

Vector3 operator* (const Vector3 & v, float f)
{
	return Vector3(
		v.x * f,
		v.y * f,
		v.z * f
	);
}

Vector3 operator* (float f, const Vector3 & v)
{
	return v * f;
}

Vector3 operator/(const Vector3 & v, float f)
{
	return Vector3(
		v.x / f,
		v.y / f,
		v.z / f
	);
}

ostream & operator<<(ostream & outputStream, const Vector3 & vector)
{
	// Bit of a hack for outputting as color from 0-255

	outputStream << int(vector.x * 255.99) << " " << int(vector.y * 255.99) << " " << int(vector.z * 255.99);
	return outputStream;
}

Vector3 randomVectorInsideUnitSphere()
{
    // @Slow
    // Better implementations exist, but this one is the easiest!

    Vector3 result;
    do
    {
        // Random vector where x, y, z are [0-1)
        result = Vector3(rand0Incl1Excl(), rand0Incl1Excl(), rand0Incl1Excl());

        // Transform x, y, z to be [-1, 1)
        result *= 2;
        result -= Vector3(1, 1, 1);

        // Reject if outside unit sphere
    } while(result.lengthSquared() > 1.0f);

    return result;
}

Vector3 reflect(const Vector3 & v, const Vector3 & n)
{
    Vector3 result = v - 2 * dot(v, n) * n;
    return result;
}

bool refractWithSchlickProbability(const Vector3 & v, const Vector3 & n, float refractionIndexFrom, float refractionIndexTo, Vector3 * vecOut)
{
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel


    Vector3 nAdjusted = n;
    if (dot(v, n) > 0)
    {
        nAdjusted = -n;
    }
    
    float cosThetaIncident = dot(-v, nAdjusted);

	// Reject due to Schlick probability

	{
		float reflectProb = (1 - refractionIndexTo) / (1 + refractionIndexTo);
		reflectProb = reflectProb * reflectProb;
		reflectProb = reflectProb + (1 - reflectProb) * pow(1 - cosThetaIncident, 5);

		if (rand0Incl1Excl() < reflectProb)
			return false;
	}

    float refractionRatio = refractionIndexFrom / refractionIndexTo;
    float k = 1 - refractionRatio * refractionRatio * (1 - cosThetaIncident * cosThetaIncident);

	// Reject due to total internal reflection

	if (k < 0)
		return false;
    
    *vecOut = refractionRatio * v + nAdjusted * (refractionRatio * cosThetaIncident - sqrtf(k));
	return true;
}

float rand0Incl1Excl()
{
	return rand() / (RAND_MAX + 1.0f);
}

bool Sphere::testHit(const Ray & ray, float tMin, float tMax, HitRecord * hitOut) const
{
	float a = dot(ray.dir, ray.dir);
	float b = 2 * (dot(ray.dir, ray.p0) - dot(ray.dir, this->center));
	float c = dot(ray.p0, ray.p0) + dot(this->center, this->center) - 2 * dot(ray.p0, this->center) - this->radius * this->radius;

	float discriminant = b * b - 4 * a * c;
	if (discriminant < 0)
		return false;

	float t0 = (-b - sqrt(discriminant)) / 2 * a;
	float t1 = (-b + sqrt(discriminant)) / 2 * a;

	if (t0 > tMin && t0 < tMax)
	{
		hitOut->t = t0;
		hitOut->normal = normalize(ray.pointAtT(t0) - this->center);
        hitOut->hitable = this;
		return true;
	}
	else if (t1 > tMin && t1 < tMax)
	{
		hitOut->t = t1;
		hitOut->normal = normalize(ray.pointAtT(t1) - this->center);
        hitOut->hitable = this;
		return true;
	}
	else
	{
		return false;
	}
}

Vector3 Ray::pointAtT(float t) const
{
	return p0 + dir * t;
}

Vector3 Ray::color(IHitable ** aHitable, int cHitable, int rayDepth) const
{
    constexpr int MAX_RAY_DEPTH = 50;
    if (rayDepth > MAX_RAY_DEPTH)
    {
        return Vector3(0, 0, 0);
    }
    
	float tClosest = FLT_MAX;
	HitRecord hitClosest;

	for (int iHitable = 0; iHitable < cHitable; iHitable++)
	{
		IHitable * pHitable = *(aHitable + iHitable);

		HitRecord hit;
		if (pHitable->testHit(*this, 0.0001, tClosest, &hit))
		{
			hitClosest = hit;
			tClosest = hit.t;
		}
	}

	bool hit = tClosest < FLT_MAX;
	if (hit)
	{
        Vector3 attenuation;
        Ray rayScattered;
        
        if (hitClosest.hitable->material->scatter(*this, hitClosest, &attenuation, &rayScattered))
        {
            return hadamard(attenuation, rayScattered.color(aHitable, cHitable, rayDepth + 1));
        }
        else
        {
            return Vector3(0, 0, 0);
        }
	}
	else
	{
		// blue-white "sky" lerp

		float t = 0.5f * (dir.y + 1);
		return (1 - t) * Vector3(1, 1, 1) + t * Vector3(0.5f, 0.7f, 1.0f);
	}
}

bool LambertianMaterial::scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const
{
    if (dot(ray.dir, hitRecord.normal) > 0)
        return false;
    
    Vector3 hitPos = ray.pointAtT(hitRecord.t);
    Vector3 probeUnitSphereCenter = hitPos + hitRecord.normal;
    *rayScatteredOut = Ray(hitPos, probeUnitSphereCenter + randomVectorInsideUnitSphere() - hitPos);
    *attenuationOut = albedo;
    return true;
}

bool MetalMaterial::scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const
{
    if (dot(ray.dir, hitRecord.normal) > 0)
        return false;
    
    Vector3 hitPos = ray.pointAtT(hitRecord.t);
    *rayScatteredOut = Ray(hitPos, reflect(ray.dir, hitRecord.normal) + fuzziness * randomVectorInsideUnitSphere());
    *attenuationOut = albedo;
    return true;
}

bool DielectricMaterial::scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const
{
	*attenuationOut = Vector3(1.0, 1.0, 1.0);

	float isInside = dot(ray.dir, hitRecord.normal) > 0;

	float refractiveIndexFrom	=	isInside	?	this->refractiveIndex	:	1.0f;
	float refractiveIndexTo		=	isInside	?	1.0f					:	this->refractiveIndex;

	Vector3 refracted;
	if (refractWithSchlickProbability(ray.dir, isInside ? -hitRecord.normal : hitRecord.normal, refractiveIndexFrom, refractiveIndexTo, &refracted))
	{
		rayScatteredOut->p0 = ray.pointAtT(hitRecord.t);
		rayScatteredOut->dir = refracted;
	}
	else
	{
		rayScatteredOut->p0 = ray.pointAtT(hitRecord.t);
		rayScatteredOut->dir = reflect(ray.dir, hitRecord.normal);
	}

	return true;
}

int main()
{
	constexpr int widthPixels = 200;
	constexpr int heightPixels = 100;

	ofstream file;
	file.open("output.ppm");
	file << "P3" << endl;
	file << widthPixels << " " << heightPixels << endl;
	file << 255 << endl;

	// Camera is at origin
	// Viewing plane is right-handed: +X right, +Y up, +Z out
	// Viewing plane is at Z = -1
	// Viewing plane spans X from -2 to 2
	// Viewing plane spans Y from -1 to 1

	Vector3 origin(0, 0, 0);
	Vector3 viewLowerLeftCorner(-2, -1, -1);
	Vector3 viewX(1, 0, 0);
	Vector3 viewY(0, 1, 0);
	float viewWidth = 4;
	float viewHeight = 2;

	constexpr int COUNT_HITABLES = 5;
	IHitable * hitables[COUNT_HITABLES];
	hitables[0] =
        new Sphere(
            Vector3(0, 0, -1),
            0.5,
            new LambertianMaterial(Vector3(0.1f, 0.2f, 0.5f)));


    hitables[1] =
        new Sphere(
            Vector3(0, -100.5f, -1),
            100,
            new LambertianMaterial(
                Vector3(0.8f, 0.8f, 0.0f)));

    hitables[2] =
        new Sphere(
            Vector3(1, 0, -1),
            0.5,
            new MetalMaterial(
                Vector3(0.8f, 0.6f, 0.2f),
				1.0f));

    hitables[3] =
        new Sphere(
            Vector3(-1, 0, -1),
            0.5,
			new DielectricMaterial(1.5f));

	hitables[4] =
		new Sphere(
			Vector3(-1, 0, -1),
			0.45,
			new DielectricMaterial(1.5f));

	for (int yPixel = 0; yPixel < heightPixels; yPixel++)
	{
		for (int xPixel = 0; xPixel < widthPixels; xPixel++)
		{
			Vector3 colorCumulative;

			constexpr int COUNT_SAMPLES = 100;
			for (int iSample = 0; iSample < COUNT_SAMPLES; iSample++)
			{
				float perturbX = rand0Incl1Excl();
				float perturbY = rand0Incl1Excl();

				float xViewNormalized = (xPixel + perturbX) / widthPixels;
				float yViewNormalized = 1 - ((yPixel + perturbY) / heightPixels);	// Subtract from 1 to get the value in view-space

				Ray ray(
					origin,
					viewLowerLeftCorner + xViewNormalized * viewX * viewWidth + yViewNormalized * viewY * viewHeight);

				colorCumulative += ray.color(hitables, COUNT_HITABLES, 0);
			}


			Vector3 colorOut = colorCumulative / COUNT_SAMPLES;

            // Gamma correct

            colorOut.x = sqrt(colorOut.x);
            colorOut.y = sqrt(colorOut.y);
            colorOut.z = sqrt(colorOut.z);
            
			file << colorOut << endl;
		}
	}

	file.close();
	return 0;
}
