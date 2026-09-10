# -*- coding: utf-8 -*-
# 说明: 原 send_mail.py 含真实邮箱授权码, 为安全起见未收录到仓库, 这里提供脱敏示例。
import smtplib, ssl, sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header

SMTP_HOST = "smtp.example.com"
SMTP_PORT = 465
SENDER    = "you@example.com"
AUTH      = "PUT-YOUR-SMTP-AUTH-CODE-HERE"   # 不要提交真实授权码
RECIPS    = ["someone@example.com"]
ZIP       = r"./package.zip"

def main():
    msg = MIMEMultipart()
    msg["From"] = SENDER
    msg["To"] = ", ".join(RECIPS)
    msg["Subject"] = Header("交付包", "utf-8")
    msg.attach(MIMEText("见附件。", "plain", "utf-8"))
    part = MIMEBase("application", "octet-stream")
    with open(ZIP, "rb") as f:
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment", filename=("utf-8", "", "package.zip"))
    msg.attach(part)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=60, context=ctx) as srv:
        srv.login(SENDER, AUTH)
        srv.sendmail(SENDER, RECIPS, msg.as_string())
    print("SEND OK")

if __name__ == "__main__":
    main()

