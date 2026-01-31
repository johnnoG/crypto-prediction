"""add_alert_and_watchlist_tables

Revision ID: aa81d8e1a021
Revises: 196bf5c0a663
Create Date: 2026-01-31 18:52:29.722349

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'aa81d8e1a021'
down_revision: Union[str, Sequence[str], None] = '196bf5c0a663'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create useralert table
    op.create_table('useralert',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('crypto_symbol', sa.String(length=20), nullable=False),
        sa.Column('crypto_name', sa.String(length=100), nullable=False),
        sa.Column('alert_type', sa.Enum('PRICE_TARGET', 'FORECAST_CHANGE', 'VOLATILITY', name='alerttype'), nullable=False),
        sa.Column('target_price', sa.Float(), nullable=True),
        sa.Column('condition', sa.String(length=20), nullable=True),
        sa.Column('status', sa.Enum('ACTIVE', 'TRIGGERED', 'CANCELLED', name='alertstatus'), nullable=False, default='ACTIVE'),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('triggered_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create userwatchlist table
    op.create_table('userwatchlist',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('crypto_symbol', sa.String(length=20), nullable=False),
        sa.Column('crypto_name', sa.String(length=100), nullable=False),
        sa.Column('crypto_id', sa.String(length=50), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('is_favorite', sa.Boolean(), nullable=False, default=False),
        sa.Column('notification_enabled', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create unique constraint for user_id + crypto_id in watchlist
    op.create_unique_constraint('uq_user_crypto_watchlist', 'userwatchlist', ['user_id', 'crypto_id'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order
    op.drop_constraint('uq_user_crypto_watchlist', 'userwatchlist', type_='unique')
    op.drop_table('userwatchlist')
    op.drop_table('useralert')

    # Drop custom enum types
    op.execute("DROP TYPE IF EXISTS alerttype")
    op.execute("DROP TYPE IF EXISTS alertstatus")
