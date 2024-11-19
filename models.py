from datetime import datetime
from app import db

class Referral(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    urgency = db.Column(db.String(50), nullable=False)  # 'urgent' or 'routine'
    appointment_type = db.Column(db.String(100), nullable=False)
    field_test_required = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Referral {self.filename}>'
