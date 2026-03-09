/*
 * ThermalOS – Smart Heat Pipe Sensor Node Firmware
 * ESP32-S3 + SX1262 LoRa (Heltec WiFi LoRa 32 V3)
 * TEG-powered via LTC3108 + 1F Supercap
 *
 * Reads 5× DS18B20 sensors, transmits via LoRa every 60s
 * Deep-sleep between measurements for minimal power consumption
 */

#include <OneWire.h>
#include <DallasTemperature.h>
#include <LoRa.h>
#include <Wire.h>
#include "esp_sleep.h"

// ── Pin Configuration ──────────────────────────────────
#define ONE_WIRE_BUS    4      // DS18B20 data pin
#define LORA_SCK        5
#define LORA_MISO       19
#define LORA_MOSI       27
#define LORA_CS         18
#define LORA_RST        14
#define LORA_IRQ        26
#define LED_PIN         25
#define VBAT_ADC        36     // Battery/Supercap voltage monitoring
#define POWER_GOOD_PIN  13     // LTC3108 Power Good indicator

// ── Configuration ──────────────────────────────────────
#define SLEEP_DURATION_US  60000000UL  // 60 seconds deep sleep
#define LORA_FREQUENCY     868E6       // EU 868 MHz
#define LORA_BANDWIDTH     125E3
#define LORA_SPREADING     7           // SF7 for minimum airtime
#define LORA_TX_POWER      14          // dBm
#define NUM_SENSORS        5
#define NODE_ID            "HP01"

// ── Sensor Positions (meters from evaporator end) ──────
const float SENSOR_POS[NUM_SENSORS] = {0.02, 0.05, 0.125, 0.20, 0.24};
const char* SENSOR_NAMES[NUM_SENSORS] = {
    "T_evap_hot", "T_evap_mid", "T_adiabatic", "T_cond_mid", "T_cond_end"
};

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

RTC_DATA_ATTR uint32_t bootCount = 0;  // Persists across deep sleep

void setup() {
    bootCount++;
    Serial.begin(115200);

    // Check power status
    pinMode(POWER_GOOD_PIN, INPUT);
    if (!digitalRead(POWER_GOOD_PIN)) {
        // Insufficient power – go back to sleep immediately
        Serial.println("[WARN] Power not ready, sleeping...");
        enterDeepSleep();
        return;
    }

    // Initialize sensors
    sensors.begin();
    sensors.setResolution(12);  // 12-bit = ±0.0625°C

    // Initialize LoRa
    LoRa.setPins(LORA_CS, LORA_RST, LORA_IRQ);
    if (!LoRa.begin(LORA_FREQUENCY)) {
        Serial.println("[ERROR] LoRa init failed!");
        enterDeepSleep();
        return;
    }
    LoRa.setSpreadingFactor(LORA_SPREADING);
    LoRa.setSignalBandwidth(LORA_BANDWIDTH);
    LoRa.setTxPower(LORA_TX_POWER);

    // Read all sensors
    sensors.requestTemperatures();
    delay(750);  // Wait for 12-bit conversion

    // Read supercap voltage
    float vSupercap = analogRead(VBAT_ADC) * (5.5 / 4095.0);

    // Build JSON payload
    String payload = "{";
    payload += "\"id\":\"" + String(NODE_ID) + "\",";
    payload += "\"boot\":" + String(bootCount) + ",";
    payload += "\"v_sc\":" + String(vSupercap, 2) + ",";
    payload += "\"T\":[";
    for (int i = 0; i < NUM_SENSORS; i++) {
        float T = sensors.getTempCByIndex(i);
        payload += String(T, 2);
        if (i < NUM_SENSORS - 1) payload += ",";
        Serial.printf("[%s] %.2f °C\n", SENSOR_NAMES[i], T);
    }
    payload += "]}";

    // Transmit via LoRa
    Serial.println("TX: " + payload);
    LoRa.beginPacket();
    LoRa.print(payload);
    LoRa.endPacket();

    // Blink LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);
    delay(50);
    digitalWrite(LED_PIN, LOW);

    Serial.printf("Boot #%d, V_sc=%.2fV, Sleeping %ds...\n",
                  bootCount, vSupercap, SLEEP_DURATION_US/1000000);

    enterDeepSleep();
}

void enterDeepSleep() {
    LoRa.sleep();
    esp_sleep_enable_timer_wakeup(SLEEP_DURATION_US);
    esp_deep_sleep_start();
}

void loop() {
    // Never reached – deep sleep after setup
}
