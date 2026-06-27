from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QSizePolicy
except Exception:  # pragma: no cover - allows non-Qt test environments to skip cleanly.
    QApplication = None  # type: ignore[assignment]
    Qt = None  # type: ignore[assignment]
    QSizePolicy = None  # type: ignore[assignment]


@unittest.skipIf(QApplication is None, "PyQt6 is not available")
class CvOpsNotificationsPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_notification_center_is_available_in_top_activity_rail(self) -> None:
        from insight_local.cvops.ui.activity_rail import ActivityRailWidget

        rail = ActivityRailWidget(orientation=Qt.Orientation.Horizontal)
        emitted: list[str] = []
        try:
            rail.modePressed.connect(emitted.append)

            button = rail._mode_buttons.get("notifications")
            self.assertIsNotNone(button)
            self.assertEqual(button.text(), "Notifications")
            self.assertIn("Notification Center", button.toolTip())

            button.click()

            self.assertEqual(emitted, ["notifications"])
            rail.set_workbench_mode("notifications")
            self.assertTrue(button.isChecked())
        finally:
            rail.deleteLater()

    def test_repeated_job_events_collapse_into_one_card(self) -> None:
        from insight_local.cvops.ui.notifications_panel import NotificationsPanel

        panel = NotificationsPanel()
        try:
            panel.ingest(
                {
                    "type": "job_status",
                    "job_id": "job-1",
                    "scenario": "demo",
                    "state": "queued",
                    "emitted_at": 1_700_000_000,
                }
            )
            panel.ingest(
                {
                    "type": "job_status",
                    "job_id": "job-1",
                    "scenario": "demo",
                    "state": "running",
                    "emitted_at": 1_700_000_001,
                }
            )
            panel.ingest(
                {
                    "type": "job_status",
                    "job_id": "job-2",
                    "scenario": "demo",
                    "state": "queued",
                    "emitted_at": 1_700_000_002,
                }
            )

            self.assertEqual(panel._group_order, ["job_status|job-2", "job_status|job-1"])
            self.assertEqual(panel._count.text(), "2 stacks · 3 notifications")

            repeated = panel._group_cards["job_status|job-1"]
            self.assertEqual(repeated._count_badge.text(), "2")
            repeated._toggle.click()
            self.assertTrue(repeated.is_expanded())
            self.assertEqual(repeated._details.count(), 2)
        finally:
            panel.deleteLater()

    def test_open_group_stays_expanded_when_related_event_arrives(self) -> None:
        from insight_local.cvops.ui.notifications_panel import NotificationsPanel

        panel = NotificationsPanel()
        try:
            panel.ingest(
                {
                    "type": "local_error",
                    "scope": "websocket",
                    "state": "error",
                    "message": "disconnected",
                    "emitted_at": 1_700_000_010,
                }
            )
            card = panel._group_cards["local_error|websocket"]
            card._toggle.click()
            self.assertTrue(card.is_expanded())

            panel.ingest(
                {
                    "type": "local_error",
                    "scope": "websocket",
                    "state": "error",
                    "message": "disconnected",
                    "emitted_at": 1_700_000_011,
                }
            )

            card = panel._group_cards["local_error|websocket"]
            self.assertTrue(card.is_expanded())
            self.assertEqual(card._count_badge.text(), "2")
            self.assertEqual(card._details.count(), 2)
        finally:
            panel.deleteLater()

    def test_notification_cards_stay_top_aligned_and_compact(self) -> None:
        from insight_local.cvops.ui.notifications_panel import NotificationsPanel

        panel = NotificationsPanel()
        try:
            panel.ingest(
                {
                    "type": "local_error",
                    "scope": "health",
                    "message": "timed out",
                    "emitted_at": 1_700_000_020,
                }
            )

            card = panel._group_cards["local_error|health"]
            self.assertEqual(card.sizePolicy().verticalPolicy(), QSizePolicy.Policy.Fixed)
            self.assertTrue(panel._cards_layout.alignment() & Qt.AlignmentFlag.AlignTop)
        finally:
            panel.deleteLater()

    def test_heartbeat_cards_only_emit_for_meaningful_changes(self) -> None:
        from insight_local.cvops.ui.notification_cards import HeartbeatNotificationGate

        gate = HeartbeatNotificationGate()

        self.assertFalse(
            gate.should_emit(
                {
                    "type": "heartbeat",
                    "state": "live",
                    "queued": 0,
                    "running": 0,
                    "error": 0,
                }
            )
        )
        self.assertFalse(
            gate.should_emit(
                {
                    "type": "heartbeat",
                    "state": "live",
                    "queued": 0,
                    "running": 0,
                    "error": 0,
                }
            )
        )
        self.assertTrue(
            gate.should_emit(
                {
                    "type": "heartbeat",
                    "state": "live",
                    "queued": 1,
                    "running": 0,
                    "error": 0,
                }
            )
        )
        self.assertFalse(
            gate.should_emit(
                {
                    "type": "heartbeat",
                    "state": "live",
                    "queued": 1,
                    "running": 0,
                    "error": 0,
                }
            )
        )
        self.assertTrue(
            gate.should_emit(
                {
                    "type": "heartbeat",
                    "state": "degraded",
                    "queued": 0,
                    "running": 0,
                    "error": 0,
                }
            )
        )
        self.assertTrue(
            gate.should_emit(
                {
                    "type": "heartbeat",
                    "state": "live",
                    "queued": 0,
                    "running": 0,
                    "error": 0,
                }
            )
        )

    def test_notification_card_tray_collapses_repeated_job_updates(self) -> None:
        from insight_local.cvops.ui.notification_cards import NotificationCardTray

        tray = NotificationCardTray(ttl_ms=60_000)
        try:
            tray.push(
                {
                    "type": "job_status",
                    "job_id": "job-1",
                    "scenario": "demo",
                    "state": "queued",
                    "emitted_at": 1_700_000_100,
                }
            )
            tray.push(
                {
                    "type": "job_status",
                    "job_id": "job-1",
                    "scenario": "demo",
                    "state": "running",
                    "emitted_at": 1_700_000_101,
                }
            )

            self.assertEqual(tray.card_count(), 1)
            self.assertEqual(tray._order, ["job_status|job-1"])
            card = tray._cards["job_status|job-1"]
            self.assertEqual(tray._counts["job_status|job-1"], 2)
            self.assertEqual(card._count_badge.text(), "2")
            self.assertIn("running", card.summary_text().lower())
        finally:
            tray.deleteLater()

    def test_notification_card_tray_can_run_as_single_overlay_card(self) -> None:
        from insight_local.cvops.ui.notification_cards import NotificationCardTray

        tray = NotificationCardTray(max_cards=1, ttl_ms=60_000)
        try:
            tray.push(
                {
                    "type": "local_error",
                    "scope": "websocket",
                    "state": "error",
                    "message": "disconnected",
                    "emitted_at": 1_700_000_200,
                }
            )
            tray.push(
                {
                    "type": "job_status",
                    "job_id": "job-2",
                    "scenario": "demo",
                    "state": "running",
                    "emitted_at": 1_700_000_201,
                }
            )

            self.assertEqual(tray.card_count(), 1)
            self.assertEqual(tray._order, ["job_status|job-2"])
            self.assertNotIn("local_error|websocket", tray._cards)
        finally:
            tray.deleteLater()


if __name__ == "__main__":
    unittest.main()
