from __future__ import annotations

from sqlalchemy import (
    Column,
    Double,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Principal(Base):
    __tablename__ = "principals"

    id = Column(String, primary_key=True)
    type = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

    sessions = relationship("AnonymousSession", back_populates="principal")
    conversations = relationship("Conversation", back_populates="principal")
    plans = relationship("Plan", back_populates="principal")


class AnonymousSession(Base):
    __tablename__ = "anonymous_sessions"

    id = Column(String, primary_key=True)
    principal_id = Column(String, ForeignKey("principals.id"), nullable=False)
    session_key = Column(String, nullable=False, unique=True)
    first_seen_at = Column(String, nullable=False)
    last_seen_at = Column(String, nullable=False)
    user_agent = Column(Text)
    ip_hash = Column(String)

    principal = relationship("Principal", back_populates="sessions")

    __table_args__ = (Index("idx_anonymous_sessions_principal_id", "principal_id"),)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True)
    principal_id = Column(String, ForeignKey("principals.id"), nullable=False)
    title = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

    principal = relationship("Principal", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")
    plans = relationship("Plan", back_populates="conversation")

    __table_args__ = (Index("idx_conversations_principal_id", "principal_id"),)


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    metadata_json = Column(Text)
    created_at = Column(String, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (Index("idx_messages_conversation_id", "conversation_id"),)


class Plan(Base):
    __tablename__ = "plans"

    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    principal_id = Column(String, ForeignKey("principals.id"), nullable=False)
    city = Column(String)
    days = Column(Integer)
    structured_json = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

    conversation = relationship("Conversation", back_populates="plans")
    principal = relationship("Principal", back_populates="plans")

    __table_args__ = (
        Index("idx_plans_conversation_id", "conversation_id"),
        Index("idx_plans_principal_id", "principal_id"),
    )


class Place(Base):
    __tablename__ = "places"

    place_id = Column(String, primary_key=True)
    name = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    city = Column(String)
    city_key = Column(String)
    district = Column(String)
    ward = Column(String)
    address = Column(Text)
    description = Column(Text)
    detail_content = Column(Text)
    list_snippet = Column(Text)
    source = Column(String)
    source_category_code = Column(String)
    destination_type = Column(String)
    item_id = Column(String)
    detail_url = Column(Text)
    website = Column(Text)
    phone = Column(String)
    planner_role = Column(String)
    primary_area_key = Column(String)
    admin_area_keys_json = Column(Text)
    intent_tags_json = Column(Text)
    density_bucket = Column(String)
    verification_status = Column(String)
    lat = Column(Double)
    lon = Column(Double)
    payload_json = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

    chunks = relationship("PlaceChunk", back_populates="place", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_places_category", "category"),
        Index("idx_places_city_key", "city_key"),
        Index("idx_places_primary_area_key", "primary_area_key"),
        Index("idx_places_source", "source"),
    )


class PlaceChunk(Base):
    __tablename__ = "place_chunks"

    doc_id = Column(String, primary_key=True)
    place_id = Column(String, ForeignKey("places.place_id", ondelete="CASCADE"), nullable=False)
    title = Column(Text)
    city = Column(String)
    category = Column(String)
    chunk_index = Column(Integer, nullable=False)
    document_text = Column(Text, nullable=False)
    metadata_json = Column(Text)
    embedding_model = Column(String)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

    place = relationship("Place", back_populates="chunks")

    __table_args__ = (
        Index("idx_place_chunks_place_id", "place_id"),
        Index("idx_place_chunks_category", "category"),
    )
