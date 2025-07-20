"""merge translation and auth heads

Revision ID: 991033eff52e
Revises: a1b2c3d4e5f6, dd9ccc2ecc68
Create Date: 2025-07-21 02:01:26.378828

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '991033eff52e'
down_revision: Union[str, Sequence[str], None] = ('a1b2c3d4e5f6', 'dd9ccc2ecc68')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
