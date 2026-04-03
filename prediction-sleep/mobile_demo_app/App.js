import React, { useMemo, useState } from "react";
import {
  SafeAreaView,
  ScrollView,
  Text,
  TextInput,
  View,
  Pressable,
  StyleSheet
} from "react-native";
import contract from "./src/contract.json";
import { predictWithLinearContract } from "./src/inference";

function buildDefaultState() {
  const initial = {};
  for (const key of contract.feature_order) {
    initial[key] = String(contract.imputer_median[key]);
  }
  return initial;
}

export default function App() {
  const [inputs, setInputs] = useState(buildDefaultState());
  const [result, setResult] = useState(null);

  const topFeatures = useMemo(
    () => [
      "total_sleep_minutes",
      "sleep_efficiency",
      "rem_pct",
      "deep_pct",
      "hr_mean",
      "hr_std",
      "hrv_mean",
      "resp_mean",
      "spo2_mean"
    ],
    []
  );

  const setField = (field, value) => {
    setInputs((prev) => ({ ...prev, [field]: value }));
  };

  const runPrediction = () => {
    const parsed = {};
    for (const [k, v] of Object.entries(inputs)) {
      parsed[k] = Number(v);
    }
    const out = predictWithLinearContract(parsed, contract);
    setResult(out);
  };

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Sleep Fatigue Mobile Demo</Text>
        <Text style={styles.subtitle}>
          On-device linear inference using exported model contract (no server).
        </Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Key Input Features</Text>
          {topFeatures.map((feature) => (
            <View key={feature} style={styles.row}>
              <Text style={styles.label}>{feature}</Text>
              <TextInput
                style={styles.input}
                keyboardType="numeric"
                value={inputs[feature]}
                onChangeText={(text) => setField(feature, text)}
              />
            </View>
          ))}

          <Pressable style={styles.button} onPress={runPrediction}>
            <Text style={styles.buttonText}>Run On-Device Prediction</Text>
          </Pressable>
        </View>

        {result && (
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>Prediction</Text>
            <Text style={styles.resultText}>Label: {String(result.label)}</Text>
            <Text style={styles.resultText}>
              P(0): {result.probabilities[0].toFixed(3)}
            </Text>
            <Text style={styles.resultText}>
              P(1): {result.probabilities[1].toFixed(3)}
            </Text>
            <Text style={styles.resultMeta}>Logit: {result.logit.toFixed(3)}</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#f3f4f6"
  },
  container: {
    padding: 16,
    gap: 14
  },
  title: {
    fontSize: 24,
    fontWeight: "800",
    color: "#111827"
  },
  subtitle: {
    color: "#374151",
    lineHeight: 20
  },
  card: {
    backgroundColor: "white",
    borderRadius: 14,
    padding: 14,
    gap: 12
  },
  cardTitle: {
    fontWeight: "700",
    fontSize: 16,
    color: "#111827"
  },
  row: {
    gap: 6
  },
  label: {
    color: "#1f2937",
    fontSize: 13
  },
  input: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
    backgroundColor: "#f9fafb"
  },
  button: {
    marginTop: 8,
    backgroundColor: "#0f766e",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center"
  },
  buttonText: {
    color: "white",
    fontWeight: "700"
  },
  resultCard: {
    backgroundColor: "#0b1320",
    borderRadius: 14,
    padding: 14,
    gap: 6
  },
  resultTitle: {
    color: "#e5e7eb",
    fontSize: 16,
    fontWeight: "700"
  },
  resultText: {
    color: "#f9fafb",
    fontSize: 14
  },
  resultMeta: {
    color: "#9ca3af",
    marginTop: 4
  }
});

