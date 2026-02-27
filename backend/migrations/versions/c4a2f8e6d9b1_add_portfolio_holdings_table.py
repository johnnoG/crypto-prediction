"""add_portfolio_holdings_table

Revision ID: c4a2f8e6d9b1
Revises: b3b13f5b7b42
Create Date: 2026-02-26

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c4a2f8e6d9b1"
down_revision: Union[str, Sequence[str], None] = "b3b13f5b7b42"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create portfolioholding table."""
    op.create_table(
        "portfolioholding",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
        sa.Column("crypto_id", sa.String(50), nullable=False),
        sa.Column("crypto_symbol", sa.String(20), nullable=False),
        sa.Column("crypto_name", sa.String(100), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("avg_buy_price", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "crypto_id", name="_user_crypto_portfolio"),
    )
    op.create_index("ix_portfolioholding_user_id", "portfolioholding", ["user_id"])


def downgrade() -> None:
    """Drop portfolioholding table."""
    op.drop_index("ix_portfolioholding_user_id", table_name="portfolioholding")
    op.drop_table("portfolioholding")
