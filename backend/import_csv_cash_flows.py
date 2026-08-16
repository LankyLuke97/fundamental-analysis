from typing import cast
from decimal import Decimal
import csv
import datetime
from pathlib import Path

from sqlmodel import select, Session

from app.db import database
from app.db.database import engine

from app.db.schema.cash_flow import CashFlow, CashFlowRead
from app.db.schema.cash_flow_category import CashFlowCategory, CashFlowCategoryRead
from app.db.schema.cash_flow_tag import CashFlowTag, CashFlowTagRead, CashFlowTagLink


data = Path("cash_flows.csv")

MAPPING = {
    "income_studentfinance": ("income", "student_finance"),
    "income_other": ("income", "other"),
    "income_options": ("income", "options"),
    "income_work": ("income", "work"),
    "groceries": ("groceries",),
    "alchohol": ("alchohol",),
    "rent": ("rent",),
    "tax": ("tax",),
    "travel": ("travel", "general"),
    "books_education": ("books", "education"),
    "books_other": ("books", "other"),
    "entertainment": ("entertainment",),
    "eatingout": ("eating_out",),
    "clothing": ("clothing",),
    "nightout": ("night_out",),
    "pub": ("pub",),
    "utilities": ("utilities",),
    "investing": ("investing",),
    "getrichquick": ("financial",),
    "sport": ("sport",),
    "other": ("other",),
    "outby": ("adjustment",),
}

with data.open(encoding="utf8", mode="r") as file_handle:
    csv_reader = csv.DictReader(file_handle, delimiter=",")
    with Session(engine) as session:
        cash_flow_categories = {}
        cash_flow_tags = {}
        zero = Decimal("0")
        for row in csv_reader:
            timestamp = datetime.datetime.strptime(row["date"], "%d/%m/%Y").astimezone()
            print("Importing:", timestamp)
            for k, v in row.items():
                if k not in MAPPING:
                    continue
                if not v:
                    continue
                value = Decimal(v)
                if value == zero:
                    continue
                category, *tags = MAPPING[k]
                if category not in cash_flow_categories:
                    db_category = CashFlowCategory(name=category)
                    session.add(db_category)
                    session.commit()
                    session.refresh(db_category)
                    cash_flow_categories[category] = db_category
                db_category = cast(CashFlowRead, cash_flow_categories[category])
                cash_flow = CashFlow(
                    category_id=db_category.id,
                    value=value,
                )
                session.add(cash_flow)
                session.commit()
                session.refresh(cash_flow)
                cash_flow = cast(CashFlowRead, cash_flow)
                for tag in tags:
                    if tag not in cash_flow_tags:
                        db_tag = CashFlowTag(name=tag)
                        session.add(db_tag)
                        session.commit()
                        session.refresh(db_tag)
                        cash_flow_tags[tag] = db_tag
                    db_tag = cast(CashFlowTagRead, cash_flow_tags[tag])
                    session.add(
                        CashFlowTagLink(
                            cash_flow_id=cash_flow.id, cash_flow_tag_id=db_tag.id
                        )
                    )

        query = select(CashFlow)
        cash_flows = list(session.exec(query).all())
        print(len(cash_flows), "added to database")
