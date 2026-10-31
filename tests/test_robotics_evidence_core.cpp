#include "robotics_evidence_core.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace robotics_evidence;

    // 1. Test Taiwan Compliance Guard (Contact & Log Sanitization)
    {
        std::string raw_text =
            "Operator log: mobile 0912-345-678, office 03-1234-5678, mail dev@example.org.";
        auto report = TaiwanComplianceGuard::audit_and_redact(raw_text);
        assert(report.pii_or_secret_detected);
        assert(report.redaction_count == 3);
        assert(report.sanitized_text.find("0912-345-678") == std::string::npos);
        assert(report.sanitized_text.find("dev@example.org") == std::string::npos);
    }

    // 2. Test SE(3) IMU-Assisted Point Cloud Deskew & Block-Temporal Anti-Leakage Split
    {
        std::vector<Point3DTimestamped> cloud = {
            {1.0, 2.0, 3.0, 0.85, 0.05},
            {4.0, -1.0, 0.5, 0.92, 0.10}
        };
        ImuMeasurement imu{0.2, 0.0, -0.1, 0.01, -0.02, 0.03};
        auto deskewed = MultiModalPerceptionProcessor::deskew_point_cloud_se3(cloud, imu);
        assert(deskewed.size() == 2);
        assert(deskewed[0].relative_time_s == 0.0);
        assert(std::isfinite(deskewed[0].x) && std::isfinite(deskewed[1].z));

        assert(MultiModalPerceptionProcessor::verify_block_temporal_split(
            {0.0, 1.0, 2.0}, {10.0, 11.0}, 5.0));
        assert(!MultiModalPerceptionProcessor::verify_block_temporal_split(
            {0.0, 1.0, 2.0}, {2.1, 11.0}, 5.0));
    }

    // 3. Test NIST CSF 2.0 Whole-Body Control Safety Governor (Torque, Thermal & Voltage)
    {
        NistWholeBodySafetyGovernor governor;
        auto now = std::chrono::steady_clock::now();

        EtherCatJointTelemetry valid_pkt{1ULL, now, 42.5, 55.0, 48.0, 0xA5A5F00DU};
        bool ok = governor.process_and_actuate(valid_pkt, 0xA5A5F00DU, now);
        assert(ok);
        assert(!governor.is_in_safe_stop());
        assert(std::abs(governor.commanded_torque_nm() - 42.5) < 1e-9);

        // Over-temperature anomaly -> triggers NIST RESPOND Safe-Torque-Off
        EtherCatJointTelemetry overtemp_pkt{2ULL, now, 30.0, 82.0, 48.0, 0xA5A5F00DU};
        bool ok2 = governor.process_and_actuate(overtemp_pkt, 0xA5A5F00DU, now);
        assert(!ok2);
        assert(governor.is_in_safe_stop());
        assert(governor.commanded_torque_nm() == 0.0);

        // NIST RECOVER when temperature cools down and bus voltage is nominal
        bool recovered = governor.attempt_recovery(60.0, 48.0);
        assert(recovered);
        assert(!governor.is_in_safe_stop());
    }

    // 4. Test Logit-Adjusted Balanced Loss
    {
        std::vector<double> logits = {2.0, 0.5, -1.0};
        std::vector<double> priors = {0.90, 0.08, 0.02};
        double loss_rare = LogitAdjustedBalancedLoss::forward_single(logits, priors, 2, 1.0);
        assert(std::isfinite(loss_rare) && loss_rare > 0.0);
    }

    std::cout << "[PASS] All NIST CSF 2.0, Taiwan Compliance, and Robotics Core tests succeeded.\n";
    return 0;
}
