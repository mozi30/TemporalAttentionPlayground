#!/usr/bin/env bash
# setup-ubuntu-vsftpd.sh
# Linux-only (Ubuntu) one-shot setup for vsftpd with passive mode and shared directory.
# Run this inside Ubuntu (or WSL Ubuntu). Does NOT touch Windows firewall/portproxy.
#
# Usage (interactive password prompt):
#   sudo bash setup-ubuntu-vsftpd.sh \
#     --user ftpuser \
#     --share /srv/ftp_share \
#     --pasv-min 50000 \
#     --pasv-max 50099 \
#     --domain minecraftwgwg.hopto.org \
#     [--enable-ftps]
#
# Notes:
# - If running in WSL, you'll still need to forward ports from Windows to WSL separately.
# - If using UFW on a full Ubuntu install, you can optionally open ports with --open-ufw.

set -euo pipefail

FTP_USER="ftpuser"
SHARE_PATH="/srv/ftp_share"
PASV_MIN=50000
PASV_MAX=50099
DOMAIN="minecraftwgwg.hopto.org"
ENABLE_FTPS=0
OPEN_UFW=0

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)        FTP_USER="$2"; shift 2;;
    --share)       SHARE_PATH="$2"; shift 2;;
    --pasv-min)    PASV_MIN="$2"; shift 2;;
    --pasv-max)    PASV_MAX="$2"; shift 2;;
    --domain)      DOMAIN="$2"; shift 2;;
    --enable-ftps) ENABLE_FTPS=1; shift 1;;
    --open-ufw)    OPEN_UFW=1; shift 1;;
    -h|--help)
      cat <<EOF
Usage:
  sudo bash $0 --user ftpuser --share /srv/ftp_share --pasv-min 50000 --pasv-max 50099 --domain example.com [--enable-ftps] [--open-ufw]
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root: sudo bash $0 [options]" >&2
  exit 1
fi

read -s -p "Enter password for Linux user '${FTP_USER}': " FTP_PASS
echo
read -s -p "Confirm password: " FTP_PASS2
echo
if [[ "$FTP_PASS" != "$FTP_PASS2" ]]; then
  echo "Passwords do not match." >&2
  exit 1
fi

echo "[1/6] Installing vsftpd..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y vsftpd openssl

echo "[2/6] Creating user and directories..."
if id -u "$FTP_USER" >/dev/null 2>&1; then
  echo "User '$FTP_USER' exists."
else
  adduser --disabled-password --gecos "" "$FTP_USER"
fi
echo "${FTP_USER}:${FTP_PASS}" | chpasswd

mkdir -p /var/run/vsftpd/empty "$SHARE_PATH"
chown -R "$FTP_USER:$FTP_USER" "$SHARE_PATH"
chmod -R 755 "$SHARE_PATH"

echo "[3/6] Writing /etc/vsftpd.conf..."
cat >/etc/vsftpd.conf <<CFG
listen=YES
listen_ipv6=NO

anonymous_enable=NO
local_enable=YES
write_enable=YES
local_umask=022

local_root=${SHARE_PATH}
chroot_local_user=YES
allow_writeable_chroot=YES

pasv_enable=YES
pasv_min_port=${PASV_MIN}
pasv_max_port=${PASV_MAX}
pasv_addr_resolve=YES
pasv_address=${DOMAIN}

xferlog_enable=YES
xferlog_std_format=YES
log_ftp_protocol=NO
use_localtime=YES

pam_service_name=vsftpd
secure_chroot_dir=/var/run/vsftpd/empty
user_sub_token=\$USER
CFG

if [[ $ENABLE_FTPS -eq 1 ]]; then
  echo "[3b/6] Enabling FTPS with self-signed certificate..."
  [[ -f /etc/ssl/certs/vsftpd.crt && -f /etc/ssl/private/vsftpd.key ]] || \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
      -subj "/CN=${DOMAIN}" \
      -keyout /etc/ssl/private/vsftpd.key \
      -out /etc/ssl/certs/vsftpd.crt
  cat >>/etc/vsftpd.conf <<SSL
ssl_enable=YES
allow_anon_ssl=NO
force_local_logins_ssl=YES
force_local_data_ssl=YES
rsa_cert_file=/etc/ssl/certs/vsftpd.crt
rsa_private_key_file=/etc/ssl/private/vsftpd.key
SSL
fi

echo "[4/6] Restarting vsftpd..."
if command -v systemctl >/dev/null 2>&1; then
  systemctl restart vsftpd || true
  systemctl enable vsftpd || true
  systemctl status vsftpd --no-pager || true
else
  service vsftpd restart || /etc/init.d/vsftpd restart || true
fi

if [[ $OPEN_UFW -eq 1 ]] && command -v ufw >/dev/null 2>&1; then
  echo "[5/6] Opening UFW ports (21, ${PASV_MIN}-${PASV_MAX})..."
  ufw allow 21/tcp || true
  ufw allow ${PASV_MIN}:${PASV_MAX}/tcp || true
  ufw reload || true
else
  echo "[5/6] Skipping UFW changes."
fi

echo "[6/6] Done."
echo
echo "Connect using:"
echo "  Host: <machine IP>"
echo "  Port: 21  (Passive mode)"
echo "  User: ${FTP_USER}"
echo "  Password: (what you entered)"
echo
echo "If running in WSL: remember to forward ports from Windows to WSL (this script does not do Windows steps)."
