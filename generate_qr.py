import qrcode

url = "https://gaming-monetization-engine-pjnvgrlwwt7nkwkqy2fo67.streamlit.app/"
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_M,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

# Membuat gambar QR Code dan menyimpannya di folder assets
img = qr.make_image(fill_color="darkblue", back_color="white")
img.save("assets/qrcode_app.png")
print("QR Code berhasil dibuat di folder assets!")