"""add translation tables

Revision ID: a1b2c3d4e5f6
Revises: cda64484c9c3
Create Date: 2025-01-21 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import func

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = 'cda64484c9c3'
branch_labels = None
depends_on = None


def upgrade():
    # Create translation_sessions table
    op.create_table('translation_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('model_type', sa.String(), nullable=False),
        sa.Column('input_mode', sa.String(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True, default=func.now()),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('session_duration', sa.Integer(), nullable=True),
        sa.Column('translations_count', sa.Integer(), nullable=True, default=0),
        sa.Column('correct_translations', sa.Integer(), nullable=True, default=0),
        sa.Column('total_confidence', sa.Float(), nullable=True, default=0.0),
        sa.Column('average_confidence', sa.Float(), nullable=True),
        sa.Column('accuracy_percentage', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_translation_sessions_id'), 'translation_sessions', ['id'], unique=False)

    # Create translation_history table
    op.create_table('translation_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('predicted_text', sa.String(), nullable=False),
        sa.Column('target_text', sa.String(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('is_correct', sa.Boolean(), nullable=True),
        sa.Column('model_used', sa.String(), nullable=False),
        sa.Column('input_mode', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True, default=func.now()),
        sa.ForeignKeyConstraint(['session_id'], ['translation_sessions.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_translation_history_id'), 'translation_history', ['id'], unique=False)


def downgrade():
    # Drop tables in reverse order
    op.drop_index(op.f('ix_translation_history_id'), table_name='translation_history')
    op.drop_table('translation_history')
    op.drop_index(op.f('ix_translation_sessions_id'), table_name='translation_sessions')
    op.drop_table('translation_sessions')
