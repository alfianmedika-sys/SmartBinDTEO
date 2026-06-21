"""
Minimal BLE servo command tester for Raspberry Pi -> ESP32-S3.

Install on Raspberry Pi:
    pip install bleak

Examples:
    python test_ble_servo.py --scan
    python test_ble_servo.py organik
    python test_ble_servo.py anorganik
    python test_ble_servo.py tengah
"""

import argparse
import asyncio
import os

from bleak import BleakClient, BleakScanner


ESP32_NAME = os.getenv("ESP32_NAME", "SmartBin-ESP32")
ESP32_ADDRESS = os.getenv("ESP32_ADDRESS", "")
SERVICE_UUID = os.getenv(
    "ESP32_SERVICE_UUID",
    "12345678-1234-1234-1234-123456789abc",
).lower()
CMD_CHAR_UUID = os.getenv(
    "ESP32_CMD_CHAR_UUID",
    "12345678-1234-1234-1234-123456789001",
)
SCAN_TIMEOUT = float(os.getenv("ESP32_BLE_SCAN_TIMEOUT", "5"))

VALID_COMMANDS = {
    "organik",
    "anorganik",
    "anorganik_styrofoam",
    "tengah",
    "netral",
}


async def scan_devices():
    print(f"[BLE] Scanning for {SCAN_TIMEOUT:.1f}s...")
    discovered = await BleakScanner.discover(timeout=SCAN_TIMEOUT, return_adv=True)

    if not discovered:
        print("[BLE] No devices found")
        return None

    esp32_address = None
    for device, advertisement in discovered.values():
        name = advertisement.local_name or device.name or "(no name)"
        service_uuids = [uuid.lower() for uuid in advertisement.service_uuids]
        print(f"[BLE] {name} | {device.address} | services={service_uuids}")
        if name == ESP32_NAME or SERVICE_UUID in service_uuids:
            esp32_address = device.address

    if esp32_address:
        print(f"[BLE] Found target {ESP32_NAME}: {esp32_address}")
    else:
        print(f"[BLE] Target {ESP32_NAME} not found")

    return esp32_address


async def find_esp32():
    if ESP32_ADDRESS:
        return ESP32_ADDRESS

    discovered = await BleakScanner.discover(timeout=SCAN_TIMEOUT, return_adv=True)
    for device, advertisement in discovered.values():
        name = advertisement.local_name or device.name
        service_uuids = [uuid.lower() for uuid in advertisement.service_uuids]
        if name == ESP32_NAME or SERVICE_UUID in service_uuids:
            return device.address

    return None


async def send_command(command):
    command = command.lower().strip()
    if command not in VALID_COMMANDS:
        raise ValueError(f"Invalid command '{command}'. Valid: {', '.join(sorted(VALID_COMMANDS))}")

    address = await find_esp32()
    if not address:
        print(f"[BLE] ESP32 not found: {ESP32_NAME}")
        return False

    print(f"[BLE] Connecting to {ESP32_NAME} at {address}...")
    async with BleakClient(address) as client:
        if not client.is_connected:
            print("[BLE] Connection failed")
            return False

        await client.write_gatt_char(CMD_CHAR_UUID, command.encode(), response=True)
        print(f"[BLE] Sent command: {command}")

    return True


def main():
    global ESP32_ADDRESS

    parser = argparse.ArgumentParser(description="Test ESP32-S3 BLE servo command.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(VALID_COMMANDS),
        help="Command to send to ESP32 servo.",
    )
    parser.add_argument(
        "--address",
        default=ESP32_ADDRESS,
        help="Connect directly to a BLE MAC/address instead of scanning by name.",
    )
    parser.add_argument("--scan", action="store_true", help="Scan nearby BLE devices.")
    args = parser.parse_args()

    if args.scan:
        asyncio.run(scan_devices())
        return

    if not args.command:
        parser.error("command is required unless --scan is used")

    if args.address:
        ESP32_ADDRESS = args.address

    ok = asyncio.run(send_command(args.command))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
