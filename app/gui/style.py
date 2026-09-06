"""ERA AI — Qt stylesheet (dark theme)."""

DARK = """
* { font-family: "Segoe UI", "Ubuntu", "Noto Sans", sans-serif; }

QMainWindow, QWidget { background: #0f1115; color: #e6e8ee; }

#Sidebar {
    background: #14171d;
    border-right: 1px solid #232733;
}
#BrandName { font-size: 18px; font-weight: 700; color: #ffffff; }
#BrandSub  { font-size: 11px; color: #7c93ff; }

QPushButton#NavBtn {
    text-align: left; padding: 10px 14px; border: none;
    border-radius: 8px; color: #b9c0cf; background: transparent;
    font-size: 13px;
}
QPushButton#NavBtn:hover { background: #1d222c; color: #ffffff; }
QPushButton#NavBtn:checked { background: #232a3a; color: #ffffff; }

#Header { background: #14171d; border-bottom: 1px solid #232733; }
#HeaderTitle { font-size: 14px; font-weight: 600; }
#StatusLine { color: #8b93a7; font-size: 12px; }

QComboBox {
    background: #1a1e27; border: 1px solid #2b3040; border-radius: 8px;
    padding: 6px 10px; color: #e6e8ee; min-width: 190px;
}
QComboBox QAbstractItemView {
    background: #1a1e27; color: #e6e8ee; selection-background-color: #2b3350;
}

QScrollArea, #ChatCanvas { background: #0f1115; border: none; }

#BubbleUser {
    background: #2b3350; border-radius: 14px; padding: 12px 14px;
    color: #ffffff;
}
#BubbleEra {
    background: #171b23; border: 1px solid #232733; border-radius: 14px;
    padding: 12px 14px; color: #e6e8ee;
}
#RoleTag { color: #7c93ff; font-size: 11px; font-weight: 700; }
#RoleTagUser { color: #9aa4bf; font-size: 11px; font-weight: 700; }
#Activity { color: #8b93a7; font-size: 12px; font-style: italic; }

QTextEdit#Composer {
    background: #14171d; border: 1px solid #2b3040; border-radius: 12px;
    padding: 10px 12px; color: #e6e8ee; font-size: 14px;
}
QPushButton#Send {
    background: #4c63d2; border: none; border-radius: 12px;
    padding: 10px 20px; color: white; font-weight: 600;
}
QPushButton#Send:hover { background: #5b72e6; }
QPushButton#Stop {
    background: #3a2030; border: 1px solid #6b2b45; border-radius: 12px;
    padding: 10px 18px; color: #ff9db5; font-weight: 600;
}
"""
