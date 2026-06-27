# Entity Panel for Displaying AI Analysis Results
# ═══════════════════════════════════════════════════════════════════════════════

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QScrollArea, QFrame, 
                             QLabel, QTextEdit, QPushButton, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import (QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, 
                         QPolygon, QFontMetrics, QPainterPath, QRadialGradient)
import time
import math

from utils.debug import debug_print


class EntityCard(QFrame):
    """Individual entity card with 3D angled folder design."""
    
    # Signal emitted when close button is clicked
    close_requested = pyqtSignal(int)  # entity_id
    
    def __init__(self, entity_id, text="", status="complete"):
        super().__init__()
        self.entity_id = entity_id
        self.status = status
        self.angle_offset = 8  # 3D depth offset in pixels
        self.corner_radius = 6
        self.shadow_blur = 3
        
        # Cyberpunk HUD animation properties
        self.animation_time = 0
        self.pulse_intensity = 0.5
        self.grid_opacity = 0.3
        self.scan_line_position = 0
        
        # Animation timer for cyberpunk effects
        self.hud_timer = QTimer(self)
        self.hud_timer.timeout.connect(self.update_hud_animation)
        self.hud_timer.start(50)  # 20 FPS animation
        
        self.init_ui()
        self.set_text(text)
        self.update_status(status)
        self.setMinimumHeight(150)  # Ensure space for 3D effect
        
    def init_ui(self):
        """Initialize the card UI with 3D folder layout."""
        self.setFrameStyle(QFrame.NoFrame)  # Custom painting
        
        # Adjust margins to account for 3D effect
        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.angle_offset + 10, 12, 10, self.angle_offset + 10)
        layout.setSpacing(6)
        
        # Header with ID, status, and close button
        header_layout = QHBoxLayout()
        
        self.id_label = QLabel(f"Analysis #{self.entity_id}")
        self.id_label.setFont(QFont("Arial", 9, QFont.Bold))
        header_layout.addWidget(self.id_label)
        
        header_layout.addStretch()
        
        self.status_label = QLabel("Complete")
        self.status_label.setFont(QFont("Arial", 8))
        header_layout.addWidget(self.status_label)
        
        # Close button (X)
        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("close_btn")
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setToolTip("Close this analysis card")
        self.close_btn.clicked.connect(lambda: self.close_requested.emit(self.entity_id))
        header_layout.addWidget(self.close_btn)
        
        layout.addLayout(header_layout)
        
        # Analysis text
        self.text_edit = QTextEdit()
        self.text_edit.setMaximumHeight(120)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Arial", 8))
        layout.addWidget(self.text_edit)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setMaximumHeight(25)
        self.copy_btn.clicked.connect(self.copy_text)
        button_layout.addWidget(self.copy_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMaximumHeight(25)
        self.clear_btn.clicked.connect(self.clear_text)
        button_layout.addWidget(self.clear_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        self.apply_card_style()
        
    def paintEvent(self, event):
        """Custom paint event for 3D folder design."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Get colors based on status
        colors = self.get_status_colors()
        
        # Draw 3D folder effect
        self.draw_3d_folder(painter, colors)
        
        super().paintEvent(event)
    
    def get_status_colors(self):
        """Get color scheme based on status."""
        if self.status == "streaming":
            return {
                'primary': QColor(57, 255, 122),     # Bright green
                'secondary': QColor(40, 180, 85),     # Darker green
                'background': QColor(45, 55, 48),     # Dark green tint
                'shadow': QColor(0, 0, 0, 120)       # Semi-transparent black
            }
        elif self.status == "complete":
            return {
                'primary': QColor(85, 85, 85),       # Gray
                'secondary': QColor(60, 60, 60),      # Darker gray
                'background': QColor(50, 50, 50),     # Dark gray
                'shadow': QColor(0, 0, 0, 100)
            }
        else:  # error
            return {
                'primary': QColor(255, 107, 57),     # Orange
                'secondary': QColor(200, 80, 40),     # Darker orange
                'background': QColor(65, 45, 40),     # Dark orange tint
                'shadow': QColor(0, 0, 0, 130)
            }
    
    def draw_3d_folder(self, painter, colors):
        """Draw the 3D angled folder effect."""
        rect = self.rect()
        
        # Define the main folder shape (angled at top-right)
        folder_rect = QRect(
            self.angle_offset, 
            0, 
            rect.width() - self.angle_offset - self.shadow_blur,
            rect.height() - self.angle_offset - self.shadow_blur
        )
        
        # Create folder tab (angled corner)
        tab_width = 40
        tab_height = 20
        
        # Draw shadow first
        shadow_rect = QRect(
            folder_rect.x() + self.shadow_blur,
            folder_rect.y() + self.shadow_blur,
            folder_rect.width(),
            folder_rect.height()
        )
        painter.setBrush(QBrush(colors['shadow']))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(shadow_rect, self.corner_radius, self.corner_radius)
        
        # Draw main folder body with gradient
        gradient = QLinearGradient(0, folder_rect.y(), 0, folder_rect.bottom())
        gradient.setColorAt(0.0, colors['background'].lighter(120))
        gradient.setColorAt(0.3, colors['background'])
        gradient.setColorAt(1.0, colors['background'].darker(110))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(colors['primary'], 2))
        painter.drawRoundedRect(folder_rect, self.corner_radius, self.corner_radius)
        
        # Draw folder tab
        tab_path = QPainterPath()
        tab_start_x = folder_rect.right() - tab_width
        tab_path.moveTo(tab_start_x, folder_rect.y())
        tab_path.lineTo(tab_start_x + tab_width - 8, folder_rect.y())
        tab_path.lineTo(tab_start_x + tab_width, folder_rect.y() + tab_height - 8)
        tab_path.lineTo(tab_start_x + tab_width, folder_rect.y() + tab_height)
        tab_path.lineTo(folder_rect.right(), folder_rect.y() + tab_height)
        tab_path.lineTo(folder_rect.right(), folder_rect.y())
        tab_path.closeSubpath()
        
        # Tab gradient
        tab_gradient = QLinearGradient(0, folder_rect.y(), 0, folder_rect.y() + tab_height)
        tab_gradient.setColorAt(0.0, colors['primary'].lighter(130))
        tab_gradient.setColorAt(1.0, colors['secondary'])
        
        painter.setBrush(QBrush(tab_gradient))
        painter.setPen(QPen(colors['primary'].darker(120), 1.5))
        painter.drawPath(tab_path)
        
        # Draw inner border highlight
        inner_rect = QRect(
            folder_rect.x() + 2,
            folder_rect.y() + 2,
            folder_rect.width() - 4,
            folder_rect.height() - 4
        )
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(colors['primary'].lighter(150), 1, Qt.SolidLine))
        painter.drawRoundedRect(inner_rect, self.corner_radius - 1, self.corner_radius - 1)
        
        # Draw corner accent lines
        accent_pen = QPen(colors['primary'].lighter(140), 1.5)
        painter.setPen(accent_pen)
        
        # Top-left corner accent
        painter.drawLine(
            folder_rect.x() + self.corner_radius,
            folder_rect.y() + 4,
            folder_rect.x() + self.corner_radius + 15,
            folder_rect.y() + 4
        )
        painter.drawLine(
            folder_rect.x() + 4,
            folder_rect.y() + self.corner_radius,
            folder_rect.x() + 4,
            folder_rect.y() + self.corner_radius + 15
        )
        
        # Bottom-right corner accent
        painter.drawLine(
            folder_rect.right() - self.corner_radius - 15,
            folder_rect.bottom() - 4,
            folder_rect.right() - self.corner_radius,
            folder_rect.bottom() - 4
        )
        painter.drawLine(
            folder_rect.right() - 4,
            folder_rect.bottom() - self.corner_radius - 15,
            folder_rect.right() - 4,
            folder_rect.bottom() - self.corner_radius
        )
        
        # Draw cyberpunk HUD overlays
        self.draw_cyberpunk_hud(painter, colors, folder_rect)
        
        painter.end()
    
    def update_hud_animation(self):
        """Update HUD animation parameters."""
        self.animation_time += 0.05
        
        # Pulse effect for streaming status
        if self.status == "streaming":
            self.pulse_intensity = 0.5 + 0.3 * math.sin(self.animation_time * 3)
        else:
            self.pulse_intensity = 0.5 + 0.1 * math.sin(self.animation_time * 0.5)
        
        # Scanning line effect
        self.scan_line_position = (self.scan_line_position + 2) % self.height()
        
        # Grid opacity breathing
        self.grid_opacity = 0.2 + 0.1 * math.sin(self.animation_time)
        
        self.update()  # Trigger repaint
    
    def draw_cyberpunk_hud(self, painter, colors, folder_rect):
        """Draw cyberpunk HUD overlay elements."""
        
        # Draw digital grid background
        self.draw_digital_grid(painter, colors, folder_rect)
        
        # Draw corner brackets
        self.draw_corner_brackets(painter, colors, folder_rect)
        
        # Draw scan lines for streaming status
        if self.status == "streaming":
            self.draw_scan_lines(painter, colors, folder_rect)
        
        # Draw data stream indicators
        self.draw_data_indicators(painter, colors, folder_rect)
    
    def draw_digital_grid(self, painter, colors, folder_rect):
        """Draw subtle digital grid pattern."""
        grid_color = QColor(colors['primary'])
        grid_color.setAlpha(int(self.grid_opacity * 255))
        painter.setPen(QPen(grid_color, 0.5))
        
        # Vertical lines
        for x in range(folder_rect.x(), folder_rect.right(), 20):
            painter.drawLine(x, folder_rect.y(), x, folder_rect.bottom())
        
        # Horizontal lines
        for y in range(folder_rect.y(), folder_rect.bottom(), 20):
            painter.drawLine(folder_rect.x(), y, folder_rect.right(), y)
    
    def draw_corner_brackets(self, painter, colors, folder_rect):
        """Draw cyberpunk-style corner brackets."""
        bracket_color = QColor(colors['primary'])
        bracket_color.setAlpha(int(self.pulse_intensity * 255))
        painter.setPen(QPen(bracket_color, 2))
        
        bracket_size = 12
        
        # Top-left bracket
        painter.drawLine(folder_rect.x(), folder_rect.y() + bracket_size, 
                        folder_rect.x(), folder_rect.y())
        painter.drawLine(folder_rect.x(), folder_rect.y(), 
                        folder_rect.x() + bracket_size, folder_rect.y())
        
        # Top-right bracket  
        painter.drawLine(folder_rect.right() - bracket_size, folder_rect.y(), 
                        folder_rect.right(), folder_rect.y())
        painter.drawLine(folder_rect.right(), folder_rect.y(), 
                        folder_rect.right(), folder_rect.y() + bracket_size)
        
        # Bottom-left bracket
        painter.drawLine(folder_rect.x(), folder_rect.bottom() - bracket_size, 
                        folder_rect.x(), folder_rect.bottom())
        painter.drawLine(folder_rect.x(), folder_rect.bottom(), 
                        folder_rect.x() + bracket_size, folder_rect.bottom())
        
        # Bottom-right bracket
        painter.drawLine(folder_rect.right() - bracket_size, folder_rect.bottom(), 
                        folder_rect.right(), folder_rect.bottom())
        painter.drawLine(folder_rect.right(), folder_rect.bottom(), 
                        folder_rect.right(), folder_rect.bottom() - bracket_size)
    
    def draw_scan_lines(self, painter, colors, folder_rect):
        """Draw animated scan lines for streaming status."""
        scan_color = QColor(colors['primary'])
        scan_color.setAlpha(120)
        
        # Create radial gradient for scan effect
        scan_gradient = QRadialGradient(folder_rect.center().x(), 
                                       self.scan_line_position, 
                                       30)
        scan_gradient.setColorAt(0.0, scan_color)
        scan_gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        
        painter.setBrush(QBrush(scan_gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRect(folder_rect.x(), 
                        max(folder_rect.y(), self.scan_line_position - 15), 
                        folder_rect.width(), 
                        30)
    
    def draw_data_indicators(self, painter, colors, folder_rect):
        """Draw small data transmission indicators."""
        indicator_color = QColor(colors['primary'])
        indicator_color.setAlpha(int(self.pulse_intensity * 200))
        painter.setPen(QPen(indicator_color, 1))
        painter.setBrush(QBrush(indicator_color))
        
        # Draw small dots along the edges
        dot_size = 2
        spacing = 8
        
        # Top edge indicators
        for i, x in enumerate(range(folder_rect.x() + spacing, 
                                   folder_rect.right(), spacing * 2)):
            if (i + int(self.animation_time * 5)) % 4 == 0:
                painter.drawEllipse(x - dot_size//2, folder_rect.y() + 2, 
                                   dot_size, dot_size)
        
        # Right edge indicators  
        for i, y in enumerate(range(folder_rect.y() + spacing, 
                                   folder_rect.bottom(), spacing * 2)):
            if (i + int(self.animation_time * 3)) % 3 == 0:
                painter.drawEllipse(folder_rect.right() - 4, y - dot_size//2, 
                                   dot_size, dot_size)
    
    def apply_card_style(self):
        """Apply styling to the 3D card components."""
        # Get colors for current status
        colors = self.get_status_colors()
        accent_color = colors['primary'].name()
        bg_color = colors['background'].name()
        
        self.setStyleSheet(f"""
            EntityCard {{
                background: transparent;
                margin: 2px;
            }}
            
            QLabel {{
                color: #ffffff;
                background: transparent;
                border: none;
                font-weight: bold;
            }}
            
            QTextEdit {{
                background-color: rgba(30, 30, 30, 200);
                color: #ffffff;
                border: 1px solid {accent_color};
                border-radius: 4px;
                padding: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9pt;
                selection-background-color: {accent_color};
            }}
            
            QTextEdit:focus {{
                border: 2px solid {accent_color};
                background-color: rgba(20, 20, 20, 220);
            }}
            
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(70, 70, 70, 200),
                                          stop:1 rgba(50, 50, 50, 200));
                color: #ffffff;
                border: 1px solid {accent_color};
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 8pt;
                font-weight: bold;
                min-height: 18px;
            }}
            
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(90, 90, 90, 220),
                                          stop:1 rgba(70, 70, 70, 220));
                border: 2px solid {accent_color};
            }}
            
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(40, 40, 40, 240),
                                          stop:1 rgba(60, 60, 60, 240));
                border: 2px solid {colors['secondary'].name()};
            }}
            
            QPushButton[objectName="close_btn"] {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(180, 60, 60, 200),
                                          stop:1 rgba(150, 40, 40, 200));
                color: #ffffff;
                border: 1px solid #ff4444;
                border-radius: 10px;
                font-size: 10pt;
                font-weight: bold;
            }}
            
            QPushButton[objectName="close_btn"]:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(220, 80, 80, 220),
                                          stop:1 rgba(180, 60, 60, 220));
                border: 2px solid #ff6666;
                color: #ffffff;
            }}
            
            QPushButton[objectName="close_btn"]:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(140, 40, 40, 240),
                                          stop:1 rgba(120, 30, 30, 240));
                border: 2px solid #cc3333;
            }}
        """)
    
    def set_text(self, text):
        """Set the analysis text."""
        self.text_edit.setPlainText(text)
        # Auto-scroll to bottom
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.End)
        self.text_edit.setTextCursor(cursor)
    
    def append_text(self, text):
        """Append text (for streaming updates)."""
        current_text = self.text_edit.toPlainText()
        self.set_text(current_text + text)
    
    def update_status(self, status):
        """Update the card status."""
        self.status = status
        if status == "streaming":
            self.status_label.setText("Streaming...")
            self.status_label.setStyleSheet("color: #39ff7a;")
        elif status == "complete":
            self.status_label.setText("Complete")
            self.status_label.setStyleSheet("color: #ffffff;")
        else:
            self.status_label.setText("Error")
            self.status_label.setStyleSheet("color: #ff6b39;")
        
        self.apply_card_style()
        self.update()  # Trigger repaint for status change
    
    def copy_text(self):
        """Copy the analysis text to clipboard."""
        try:
            from PyQt5.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(self.text_edit.toPlainText())
            debug_print(f"[ENTITY_CARD] Copied text for entity {self.entity_id}")
        except Exception as e:
            debug_print(f"[ENTITY_CARD] Error copying text: {e}")
    
    def clear_text(self):
        """Clear the analysis text."""
        self.text_edit.clear()
    
    def cleanup(self):
        """Cleanup animation timer."""
        if hasattr(self, 'hud_timer'):
            self.hud_timer.stop()


class EntityPanel(QWidget):
    """Panel for displaying multiple entity cards."""
    
    def __init__(self):
        super().__init__()
        self.entity_cards = {}  # entity_id -> EntityCard
        self.max_cards = 10  # Maximum number of cards to keep
        
        self.init_ui()
        debug_print("[ENTITY_PANEL] Initialized")
        
    def update_title_count(self):
        """Update the title to show current card count."""
        count = len(self.entity_cards)
        self.title_label.setText(f"AI Analysis Results ({count})")
    
    def init_ui(self):
        """Initialize the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title with card count
        self.title_label = QLabel("AI Analysis Results (0)")
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.title_label.setStyleSheet("color: #ffffff; padding: 5px;")
        layout.addWidget(self.title_label)
        
        # Scroll area for cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container widget for cards
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(5)
        self.cards_layout.addStretch()  # Push cards to top
        
        self.scroll_area.setWidget(self.cards_container)
        layout.addWidget(self.scroll_area)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_cards)
        button_layout.addWidget(self.clear_all_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        self.apply_panel_style()
    
    def apply_panel_style(self):
        """Apply cyberpunk styling to the panel."""
        self.setStyleSheet("""
            EntityPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 rgba(20, 20, 25, 255),
                                          stop:0.5 rgba(25, 25, 30, 255),
                                          stop:1 rgba(20, 20, 25, 255));
                border: 2px solid #39ff7a;
                border-radius: 8px;
            }
            
            QLabel {
                color: #39ff7a;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                text-shadow: 0 0 10px #39ff7a;
                background: transparent;
                border: none;
                padding: 8px;
            }
            
            QScrollArea {
                background: transparent;
                border: 1px solid rgba(57, 255, 122, 100);
                border-radius: 5px;
                margin: 5px;
            }
            
            QScrollArea QScrollBar:vertical {
                background: rgba(30, 30, 35, 200);
                width: 12px;
                border-radius: 6px;
                border: 1px solid #39ff7a;
            }
            
            QScrollArea QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #39ff7a,
                                          stop:1 #2ecc71);
                border-radius: 5px;
                min-height: 20px;
            }
            
            QScrollArea QScrollBar::handle:vertical:hover {
                background: #39ff7a;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(57, 255, 122, 150),
                                          stop:1 rgba(46, 204, 113, 150));
                color: #000000;
                border: 2px solid #39ff7a;
                border-radius: 6px;
                padding: 8px 16px;
                min-width: 80px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                font-size: 9pt;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(57, 255, 122, 200),
                                          stop:1 rgba(46, 204, 113, 200));
                box-shadow: 0 0 15px #39ff7a;
                border: 2px solid #ffffff;
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 rgba(46, 204, 113, 220),
                                          stop:1 rgba(57, 255, 122, 220));
                border: 2px solid #2ecc71;
            }
        """)
    
    def add_entity(self, entity_id, text="", status="complete"):
        """Add a new entity card."""
        if entity_id in self.entity_cards:
            # Update existing card
            self.update_entity(entity_id, text, status)
            return
        
        # Remove excess cards if at maximum
        if len(self.entity_cards) >= self.max_cards:
            self.remove_oldest_card()
        
        # Create new card
        card = EntityCard(entity_id, text, status)
        card.close_requested.connect(self.remove_entity)
        self.entity_cards[entity_id] = card
        
        # Insert at the beginning (top)
        self.cards_layout.insertWidget(0, card)
        
        # Scroll to top to show new card
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(0))
        
        self.update_title_count()
        debug_print(f"[ENTITY_PANEL] Added entity card {entity_id}")
    
    def update_entity(self, entity_id, text, status="streaming"):
        """Update an existing entity card."""
        if entity_id not in self.entity_cards:
            self.add_entity(entity_id, text, status)
            return
        
        card = self.entity_cards[entity_id]
        if status == "streaming":
            card.append_text(text)
        else:
            card.set_text(text)
        card.update_status(status)
        
        debug_print(f"[ENTITY_PANEL] Updated entity card {entity_id}")
    
    def remove_entity(self, entity_id):
        """Remove an entity card."""
        if entity_id in self.entity_cards:
            card = self.entity_cards[entity_id]
            card.cleanup()  # Stop animations
            self.cards_layout.removeWidget(card)
            card.deleteLater()
            del self.entity_cards[entity_id]
            self.update_title_count()
            debug_print(f"[ENTITY_PANEL] Removed entity card {entity_id}")
    
    def remove_oldest_card(self):
        """Remove the oldest entity card."""
        if not self.entity_cards:
            return
        
        # Find the oldest card (lowest entity_id)
        oldest_id = min(self.entity_cards.keys())
        self.remove_entity(oldest_id)
    
    def clear_all_cards(self):
        """Clear all entity cards with confirmation dialog."""
        if not self.entity_cards:
            return
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self, 
            "Clear All Cards",
            f"Are you sure you want to close all {len(self.entity_cards)} analysis cards?\\n\\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for entity_id in list(self.entity_cards.keys()):
                self.remove_entity(entity_id)
            debug_print("[ENTITY_PANEL] Cleared all entity cards")
    
    def get_card_count(self):
        """Get the number of entity cards."""
        return len(self.entity_cards)