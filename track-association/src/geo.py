"""Small geodesy helpers. Flat local approximations are fine for gaps of < ~400 km."""
import math

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """0 = north, 90 = east."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def to_local_km(lat0, lon0, lat, lon):
    """(north_km, east_km) of (lat, lon) relative to (lat0, lon0)."""
    n = math.radians(lat - lat0) * EARTH_R_KM
    e = math.radians(lon - lon0) * EARTH_R_KM * math.cos(math.radians(lat0))
    return n, e


def from_local_km(lat0, lon0, n, e):
    lat = lat0 + math.degrees(n / EARTH_R_KM)
    lon = lon0 + math.degrees(e / (EARTH_R_KM * max(math.cos(math.radians(lat0)), 1e-6)))
    return lat, lon


def project(lat, lon, v_north_kmh, v_east_kmh, hours):
    """Straight-line constant-velocity projection (negative hours = backwards)."""
    return from_local_km(lat, lon, v_north_kmh * hours, v_east_kmh * hours)


def heading_from_velocity(vn, ve):
    return (math.degrees(math.atan2(ve, vn)) + 360) % 360
