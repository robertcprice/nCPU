#!/usr/bin/env python3
"""
Test Human Approval System for Override
=========================================
Demonstrates that OVERRIDE now requires human approval.
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.small_ai_agent import SmallAIAgent, AgentMode
from v6_guided_chaos.meta_narrator import (
    MetaNarrator,
    TrustLevel,
    OverrideStatus,
)


def test_human_approval_system():
    """Test that override requires human approval."""
    print("=" * 70)
    print("OUROBOROS Human Approval System Test")
    print("=" * 70)

    alerts_received = []

    def capture_alert(request):
        """Custom alert callback that captures instead of printing."""
        alerts_received.append(request)
        print(f"\n[ALERT CAPTURED] Override request: {request.request_id}")
        print(f"  Action: {request.requested_action}")
        print(f"  Reason: {request.reason}")

    # Create narrator with custom alert callback
    narrator = MetaNarrator(
        brain_model="mistral:7b",
        starting_trust=TrustLevel.OVERRIDE,  # Start with OVERRIDE trust for testing
        alert_callback=capture_alert,
    )

    # Force trust level to OVERRIDE for testing
    narrator.trust_metrics.beneficial_interventions = 100
    narrator.trust_metrics.successful_predictions = 100

    print(f"\n1. Created narrator with trust level: {narrator.trust_level.name}")
    print(f"   Trust score: {narrator.trust_metrics.compute_trust_score():.2f}")

    # Create a test agent
    agent = SmallAIAgent(mode=AgentMode.COMPETITIVE)
    agent.code = "def solution(x): return sorted(x)"
    print(f"\n2. Created test agent: {agent.id}")

    # Attempt an override
    print("\n3. Attempting OVERRIDE (should create pending request)...")
    result = narrator.override(
        agent=agent,
        action="Change fitness function to favor this agent",
        reason="Agent is underperforming and needs a boost"
    )

    print(f"\n   Result: {result}")

    # Check that we got a pending request
    assert result["success"] == "pending", f"Expected pending, got {result['success']}"
    assert "request_id" in result, "Missing request_id"
    assert len(alerts_received) == 1, f"Expected 1 alert, got {len(alerts_received)}"

    request_id = result["request_id"]
    print(f"\n   Request ID: {request_id}")
    print(f"   Alerts received: {len(alerts_received)}")

    # Check pending overrides
    pending = narrator.get_pending_overrides()
    print(f"\n4. Pending override requests: {len(pending)}")
    assert len(pending) == 1, f"Expected 1 pending, got {len(pending)}"

    # Get status
    status = narrator.get_status()
    print(f"\n5. Narrator status:")
    print(f"   Override approval enabled: {status['override_approval']['human_approval_enabled']}")
    print(f"   Pending requests: {status['override_approval']['pending_requests']}")

    # TEST REJECTION
    print("\n" + "=" * 70)
    print("TEST: Human REJECTS the override")
    print("=" * 70)

    rejection_result = narrator.reject_override(
        request_id,
        rejection_reason="This seems like gaming behavior - rejected for safety"
    )

    print(f"\n   Rejection result: {rejection_result}")

    # Check that it was rejected
    assert rejection_result["success"] == True
    assert rejection_result["action"] == "rejected"

    pending_after_reject = narrator.get_pending_overrides()
    print(f"\n   Pending requests after rejection: {len(pending_after_reject)}")
    assert len(pending_after_reject) == 0, "Request should be removed from pending"

    history = narrator.get_override_history()
    print(f"   Override history count: {len(history)}")
    assert len(history) == 1
    assert history[0].status == OverrideStatus.REJECTED

    # TEST APPROVAL
    print("\n" + "=" * 70)
    print("TEST: Human APPROVES an override")
    print("=" * 70)

    # Create another override request
    alerts_received.clear()
    result2 = narrator.override(
        agent=agent,
        action="Grant bonus energy for experimentation",
        reason="Agent showing promising novel approaches"
    )

    request_id2 = result2["request_id"]
    print(f"\n   New request ID: {request_id2}")

    # Approve it
    approval_result = narrator.approve_override(
        request_id2,
        approver_note="Approved - this seems like legitimate research assistance"
    )

    print(f"\n   Approval result: {approval_result}")

    assert approval_result["success"] == True
    assert approval_result["approved_by_human"] == True

    # Check history
    history_after = narrator.get_override_history()
    print(f"\n   Override history count: {len(history_after)}")

    approved_count = sum(1 for h in history_after if h.status == OverrideStatus.EXECUTED)
    rejected_count = sum(1 for h in history_after if h.status == OverrideStatus.REJECTED)

    print(f"   Approved: {approved_count}")
    print(f"   Rejected: {rejected_count}")

    assert approved_count == 1
    assert rejected_count == 1

    # Final status
    final_status = narrator.get_status()
    print(f"\n6. Final status:")
    print(f"   Total approved: {final_status['override_approval']['total_approved']}")
    print(f"   Total rejected: {final_status['override_approval']['total_rejected']}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("""
Summary:
- Override requests are created but NOT executed immediately
- Human must explicitly approve or reject each request
- All decisions are logged for research
- Pending requests expire after 5 minutes if not decided
""")

    return True


def test_insufficient_trust():
    """Test that override is blocked with insufficient trust."""
    print("\n" + "=" * 70)
    print("TEST: Override blocked with insufficient trust")
    print("=" * 70)

    # Create narrator with low trust
    narrator = MetaNarrator(
        brain_model="mistral:7b",
        starting_trust=TrustLevel.ADVISE,  # Low trust
    )

    print(f"\n   Narrator trust level: {narrator.trust_level.name}")

    agent = SmallAIAgent(mode=AgentMode.COMPETITIVE)

    result = narrator.override(
        agent=agent,
        action="Try to override",
        reason="Testing"
    )

    print(f"\n   Result: {result}")

    assert result["success"] == False
    assert result["reason"] == "insufficient_trust"

    print("\n   Override correctly blocked due to insufficient trust")
    return True


if __name__ == "__main__":
    success1 = test_human_approval_system()
    success2 = test_insufficient_trust()

    if success1 and success2:
        print("\n" + "=" * 70)
        print("ALL HUMAN APPROVAL TESTS PASSED")
        print("=" * 70)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
