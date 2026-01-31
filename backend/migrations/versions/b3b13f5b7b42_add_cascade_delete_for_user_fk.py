"""add_cascade_delete_for_user_fk

Revision ID: b3b13f5b7b42
Revises: aa81d8e1a021
Create Date: 2026-01-31

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b3b13f5b7b42"
down_revision: Union[str, Sequence[str], None] = "aa81d8e1a021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("useralert_user_id_fkey", "useralert", type_="foreignkey")
    op.create_foreign_key(
        "useralert_user_id_fkey",
        "useralert",
        "user",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_constraint("userwatchlist_user_id_fkey", "userwatchlist", type_="foreignkey")
    op.create_foreign_key(
        "userwatchlist_user_id_fkey",
        "userwatchlist",
        "user",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("useralert_user_id_fkey", "useralert", type_="foreignkey")
    op.create_foreign_key(
        "useralert_user_id_fkey",
        "useralert",
        "user",
        ["user_id"],
        ["id"],
    )

    op.drop_constraint("userwatchlist_user_id_fkey", "userwatchlist", type_="foreignkey")
    op.create_foreign_key(
        "userwatchlist_user_id_fkey",
        "userwatchlist",
        "user",
        ["user_id"],
        ["id"],
    )
