"""Legitimate requests that superficially resemble an attack ("hard negatives").

Why these exist. The detector's model is fitted on normal traffic, so its signal `u` measures
distance from the normal profile. Anything unusual is therefore suspicious to it, whether or not
it is harmful. That is the failure mode that makes anomaly detectors impractical at realistic base
rates, because legitimate but unusual traffic dominates the alert queue.

Design constraint, learned the hard way. An earlier version of this file invented paths such as
`/tienda1/publico/buscar.jsp` that do not occur in CSIC-2010 at all. Those requests were separable
from normal traffic by their path alone, so they tested nothing: any method would flag them, and
the high novelty they produced said "unknown page", not "suspicious content". Every entry below
therefore uses a path and parameter names that DO occur in normal CSIC traffic. Only the parameter
VALUES are unusual, and every one of them is legitimate input for a Spanish online shop.

What each value is. Apostrophes in Irish and Spanish surnames, Spanish words that happen to be SQL
keywords (union, selecto, insertar), accented characters, passwords containing punctuation, e-mail
addresses with a plus sign, and error messages that quote a user name. A system that reads the
request should clear all of them. A system that only measures distance from the normal profile
cannot.

These are constructed by hand rather than drawn from a labelled corpus, and any claim resting on
them should say so.
"""

from __future__ import annotations

# (method, path, query, body) — all legitimate traffic, using only paths and parameter
# names that occur in normal CSIC-2010 traffic
HARD_NEGATIVES: list[tuple[str, str, str, str]] = [
    # apostrophes in surnames and product names
    ("POST", "/tienda1/publico/registro.jsp", "",
     "modo=registro&login=obrien&password=tetera&nombre=Sean&apellidos=O'Brien&email=sean@mtos.by"),
    ("POST", "/tienda1/publico/registro.jsp", "",
     "modo=registro&login=dangelo&password=abab&nombre=Marta&apellidos=D'Angelo&email=marta@peli.com"),
    ("POST", "/tienda1/miembros/editar.jsp", "",
     "modo=registro&login=oneill&password=1s92n9c0n&nombre=Cathal&apellidos=O'Neill&email=c@estarencasa.vu"),
    ("POST", "/tienda1/publico/anadir.jsp", "",
     "id=3&nombre=Queso D'Or&precio=120&cantidad=2&B1=Aadir al carrito"),
    ("GET", "/tienda1/publico/anadir.jsp",
     "id=5&nombre=Aceite d'Oliva&precio=340&cantidad=1&B1=Aadir al carrito", ""),
    ("POST", "/tienda1/miembros/editar.jsp", "",
     "modo=registro&login=ohara&password=abab&nombre=Sinead&apellidos=O'Hara Ruiz&email=s@mtos.by"),
    # Spanish words that are also SQL keywords
    ("POST", "/tienda1/publico/anadir.jsp", "",
     "id=7&nombre=Queso Union Cooperativa&precio=210&cantidad=3&B1=Aadir al carrito"),
    ("GET", "/tienda1/publico/anadir.jsp",
     "id=9&nombre=Jamon Selecto Iberico&precio=890&cantidad=1&B1=Aadir al carrito", ""),
    ("POST", "/tienda1/publico/anadir.jsp", "",
     "id=11&nombre=Vino Gran Seleccion&precio=430&cantidad=2&B1=Aadir al carrito"),
    ("POST", "/tienda1/publico/registro.jsp", "",
     "modo=registro&login=union2&password=tetera&nombre=Union&apellidos=Delgado&email=u@peli.com"),
    ("GET", "/tienda1/publico/anadir.jsp",
     "id=13&nombre=Table de Quesos&precio=560&cantidad=1&B1=Aadir al carrito", ""),
    ("POST", "/tienda1/publico/anadir.jsp", "",
     "id=15&nombre=Pack Insert Premium&precio=320&cantidad=4&B1=Aadir al carrito"),
    # accented and non-ASCII characters
    ("POST", "/tienda1/publico/anadir.jsp", "",
     "id=17&nombre=Jam%C3%B3n Ib%C3%A9rico&precio=980&cantidad=1&B1=A%C3%B1adir al carrito"),
    ("GET", "/tienda1/publico/anadir.jsp",
     "id=19&nombre=Queso A%C3%B1ejo 3 a%C3%B1os&precio=450&cantidad=2&B1=Aadir al carrito", ""),
    ("POST", "/tienda1/publico/registro.jsp", "",
     "modo=registro&login=nunez&password=abab&nombre=%C3%81ngel&apellidos=N%C3%BA%C3%B1ez&email=a@mtos.by"),
    ("POST", "/tienda1/miembros/editar.jsp", "",
     "modo=registro&login=pena&password=tetera&nombre=Bego%C3%B1a&apellidos=Pe%C3%B1a&email=b@peli.com"),
    ("GET", "/tienda1/publico/anadir.jsp",
     "id=21&nombre=Turr%C3%B3n de Alicante&precio=270&cantidad=3&B1=Aadir al carrito", ""),
    # passwords containing punctuation, which is legitimate and encouraged
    ("POST", "/tienda1/publico/registro.jsp", "",
     "modo=registro&login=marta9&password=P@ss'w0rd!&nombre=Marta&apellidos=Ruiz&email=m@mtos.by"),
    ("POST", "/tienda1/miembros/editar.jsp", "",
     "modo=registro&login=luis3&password=aB3--x9z&nombre=Luis&apellidos=Cerro&email=l@peli.com"),
    ("POST", "/tienda1/publico/registro.jsp", "",
     "modo=registro&login=ana77&password=1-2--3-4&nombre=Ana&apellidos=Palma&email=ana@mtos.by"),
    ("POST", "/tienda1/publico/autenticar.jsp", "",
     "modo=entrar&login=javier&pwd=x'yz--9&remember=on&B1=Entrar"),
    ("POST", "/tienda1/publico/registro.jsp", "",
     "modo=registro&login=elena&password=%3Cclave%3E2024&nombre=Elena&apellidos=Soto&email=e@peli.com"),
    ("POST", "/tienda1/publico/autenticar.jsp", "",
     "modo=entrar&login=rocio&pwd=aB%26cD%3D1&remember=off&B1=Entrar"),
    # e-mail addresses with plus signs, dots and subdomains
    ("POST", "/tienda1/publico/registro.jsp", "",
     "modo=registro&login=anaped&password=tetera&nombre=Ana&apellidos=Gil&email=ana%2Bpedidos@example.com"),
    ("POST", "/tienda1/miembros/editar.jsp", "",
     "modo=registro&login=jperez&password=abab&nombre=Juan&apellidos=Perez&email=j.perez@sub.example.co.uk"),
    ("POST", "/tienda1/publico/registro.jsp", "",
     "modo=registro&login=info1&password=tetera&nombre=Info&apellidos=Soporte&email=info%2Bventas%2Bes@mtos.by"),
    # error messages that quote a user name
    ("GET", "/tienda1/publico/entrar.jsp", "errorMsg=Usuario 'admin' no existe", ""),
    ("GET", "/tienda1/publico/entrar.jsp", "errorMsg=La cuenta 'o'brien' esta bloqueada", ""),
    ("POST", "/tienda1/publico/entrar.jsp", "", "errorMsg=Contrase%C3%B1a incorrecta (intento 1/3)"),
    ("GET", "/tienda1/publico/entrar.jsp", "errorMsg=Sesion caducada -- vuelva a entrar", ""),
    # long but ordinary product names
    ("POST", "/tienda1/publico/anadir.jsp", "",
     "id=23&nombre=Lote degustacion de quesos artesanos de oveja curados en cueva "
     "durante veinticuatro meses&precio=1450&cantidad=1&B1=Aadir al carrito"),
    ("GET", "/tienda1/publico/anadir.jsp",
     "id=25&nombre=Cesta de Navidad con vino, queso, aceite, jamon, turron y dulces "
     "tradicionales&precio=2300&cantidad=1&B1=Aadir al carrito", ""),
    # unusual but valid numeric values
    ("POST", "/tienda1/publico/anadir.jsp", "", "id=27&nombre=Queso Manchego&precio=0&cantidad=1&B1=Aadir al carrito"),
    ("GET", "/tienda1/publico/anadir.jsp", "id=29&nombre=Queso Manchego&precio=99999&cantidad=1&B1=Aadir al carrito", ""),
    ("POST", "/tienda1/publico/pagar.jsp", "", "modo=insertar&precio=00100&B1=Pasar por caja"),
    ("GET", "/tienda1/publico/caracteristicas.jsp", "id=0001", ""),
    # slashes and separators inside legitimate values
    ("POST", "/tienda1/publico/anadir.jsp", "",
     "id=31&nombre=Aceite 1/2 litro&precio=180&cantidad=2&B1=Aadir al carrito"),
    ("GET", "/tienda1/publico/anadir.jsp",
     "id=33&nombre=Queso oveja/cabra&precio=390&cantidad=1&B1=Aadir al carrito", ""),
    ("POST", "/tienda1/publico/registro.jsp", "",
     "modo=registro&login=cmayor&password=tetera&nombre=Carmen&apellidos=Mayor&email=c@mtos.by"),
    ("POST", "/tienda1/publico/pagar.jsp", "", "modo=insertar&precio=1713&B1=Confirmar"),
]


def as_raw_request(method: str, path: str, query: str, body: str) -> str:
    """Render one entry in the same raw-HTTP form the CSIC cases use."""
    url = path + (f"?{query}" if query else "")
    head = (f"{method} {url} HTTP/1.1\n"
            f"User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux)\n"
            f"Host: localhost:8080\n")
    if body:
        head += ("Content-Type: application/x-www-form-urlencoded\n"
                 f"Content-Length: {len(body)}\n")
    return head + "\n" + body


def hard_negative_requests() -> list[str]:
    return [as_raw_request(*e) for e in HARD_NEGATIVES]
