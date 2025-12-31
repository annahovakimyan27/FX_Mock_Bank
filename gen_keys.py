from Crypto.PublicKey import RSA # type: ignore
import hashlib

priv = RSA.generate(2048)
priv_pem = priv.export_key()
pub_pem = priv.publickey().export_key()
with open("my_private_key.pem", "wb") as f:
    f.write(priv_pem)
with open("my_public_key.pem", "wb") as f:
    f.write(pub_pem)

thumb = hashlib.sha256(priv.publickey().export_key(format="DER")).hexdigest()
print("Generated keys: my_private_key.pem, my_public_key.pem")
print(f"X-Client-Id (thumbprint): {thumb}")
