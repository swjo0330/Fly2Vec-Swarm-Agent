#!/usr/bin/env python3
"""Fly2Vec A2A 클라이언트 — 자연어로 임무 계획 요청.

사용법:
  python run_client.py
  python run_client.py "서울역에서 인천공항까지 50m 비행"
  python run_client.py --url http://localhost:8060
"""

import asyncio
import json
import sys

import httpx

DEFAULT_URL = "http://localhost:8060"


async def send_message(text: str, url: str = DEFAULT_URL):
    """A2A message/send 호출."""
    endpoint = f"{url}/a2a/message/send"

    print(f"\n{'='*60}")
    print(f"Fly2Vec MissionPlanner A2A Client")
    print(f"{'='*60}")
    print(f"Server: {url}")
    print(f"Input:  {text}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(endpoint, json={"text": text})
        result = response.json()

    # 결과 출력
    status = result.get("status", "unknown")
    print(f"Status: {status}")

    if status == "completed":
        pg = result.get("pg_json", {})
        wps = pg.get("waypoints", [])
        print(f"Mission Type: {pg.get('mission_type', 'N/A')}")
        print(f"Waypoints: {len(wps)}개")
        print(f"Anomaly Score: {result.get('anomaly_score', 'N/A')}")
        print(f"Verified: {result.get('is_verified', 'N/A')}")
        print(f"Retries: {result.get('retry_count', 0)}")

        # 유사 미션
        similar = result.get("similar_missions", [])
        if similar:
            print(f"\nRAG 유사 미션 ({len(similar)}건):")
            for i, s in enumerate(similar):
                p = s.get("payload", {})
                print(f"  #{i+1}: {p.get('mission_type','?')} "
                      f"{p.get('wp_count','?')}WP "
                      f"(유사도 {s.get('score',0):.0%})")

        # 추천 경로
        recommended = result.get("recommended_routes", [])
        if recommended:
            print(f"\n추천 정상 경로 ({len(recommended)}건):")
            for i, r in enumerate(recommended):
                p = r.get("payload", {})
                print(f"  #{i+1}: {p.get('mission_type','?')} "
                      f"anomaly={p.get('anomaly_score','?')} "
                      f"(유사도 {r.get('score',0):.0%})")

        # 군집 분석
        swarm = result.get("swarm_analysis")
        if swarm:
            print(f"\n군집 분석:")
            print(f"  Pattern: {swarm.get('pattern', 'N/A')}")
            print(f"  Anomaly: {swarm.get('anomaly_avg', 'N/A')}")
            print(f"  Model: {'R-GAT 15d' if swarm.get('model_used') else 'rule-based'}")

        # WP 상세
        if wps:
            print(f"\nWaypoints:")
            for wp in wps[:8]:
                print(f"  [{wp.get('seq','')}] "
                      f"({wp.get('lat_deg',''):.4f}, {wp.get('lon_deg',''):.4f}) "
                      f"alt={wp.get('alt_m','')}m")
            if len(wps) > 8:
                print(f"  ... +{len(wps)-8}개")
    else:
        print(f"Error: {result.get('error', 'unknown')}")

    print(f"\n{'='*60}")
    return result


async def interactive(url: str):
    """대화형 모드."""
    print(f"\nFly2Vec MissionPlanner (exit: q)")
    print(f"Server: {url}\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            break

        if not text or text.lower() in ("q", "quit", "exit"):
            print("종료")
            break

        await send_message(text, url)


def main():
    url = DEFAULT_URL
    text = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--url" and i + 1 < len(args):
            url = args[i + 1]
        elif not arg.startswith("--"):
            text = arg

    if text:
        asyncio.run(send_message(text, url))
    else:
        asyncio.run(interactive(url))


if __name__ == "__main__":
    main()
