"""
Servo MG996R - Auto Loop di GPIO 13 (Pin 33)
=============================================
Wiring:
  Pin 33 (GPIO 13) --> Signal servo (oranye/kuning)
  GND Raspberry Pi --> GND servo (coklat/hitam)  [common ground dengan PSU]
  PSU Eksternal 5V --> VCC servo (merah)

Cara jalankan:
  python3 servo_loop.py

Stop program: Ctrl+C
"""

import RPi.GPIO as GPIO
import time

# ─── Konfigurasi ──────────────────────────────────────────
SERVO_PIN   = 13      # GPIO 13 = Pin fisik 33
PWM_FREQ    = 50      # 50 Hz (standar servo)

MIN_ANGLE   = 0       # derajat minimum
MAX_ANGLE   = 180     # derajat maksimum
STEP        = 5       # besar langkah per gerakan (derajat)
DELAY       = 0.03    # jeda antar langkah (detik) — kurangi = lebih cepat

# ─── Helper: konversi sudut ke duty cycle ─────────────────
def angle_to_duty(angle: float) -> float:
    """
    Mapping sudut 0–180° ke duty cycle 2.5–12.5%
    (sesuai spesifikasi servo standar 1ms–2ms pulsewidth @ 50Hz)
    """
    return 2.5 + (angle / 180.0) * 10.0

# ─── Setup GPIO ───────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
pwm.start(angle_to_duty(MIN_ANGLE))
time.sleep(0.5)

print("=" * 45)
print("  Servo MG996R — Auto Loop Aktif")
print(f"  Pin  : GPIO {SERVO_PIN} (Pin fisik 33)")
print(f"  Range: {MIN_ANGLE}° → {MAX_ANGLE}°")
print(f"  Step : {STEP}°  |  Delay: {DELAY}s per step")
print("  Tekan Ctrl+C untuk berhenti")
print("=" * 45)

# ─── Loop Utama ───────────────────────────────────────────
try:
    loop_count = 0
    while True:
        loop_count += 1

        # Sweep: 0° → 180°
        print(f"\n[Loop #{loop_count}] Sweep  0° ──► 180°")
        for angle in range(MIN_ANGLE, MAX_ANGLE + 1, STEP):
            pwm.ChangeDutyCycle(angle_to_duty(angle))
            time.sleep(DELAY)

        # Tahan sebentar di posisi akhir
        time.sleep(0.3)
        pwm.ChangeDutyCycle(0)   # matikan sinyal (hindari jitter)
        time.sleep(0.2)

        # Sweep: 180° → 0°
        print(f"[Loop #{loop_count}] Sweep 180° ──► 0°")
        for angle in range(MAX_ANGLE, MIN_ANGLE - 1, -STEP):
            pwm.ChangeDutyCycle(angle_to_duty(angle))
            time.sleep(DELAY)

        # Tahan sebentar di posisi awal
        time.sleep(0.3)
        pwm.ChangeDutyCycle(0)
        time.sleep(0.2)

# ─── Cleanup saat Ctrl+C ──────────────────────────────────
except KeyboardInterrupt:
    print("\n\nProgram dihentikan. Cleanup GPIO...")

finally:
    pwm.stop()
    GPIO.cleanup()
    print("GPIO bersih. Sampai jumpa!")