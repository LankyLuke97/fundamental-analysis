def test_get_cash_flows(client):
    response = client.get("/cash_flows")
    assert response.status_code == 200
    assert response.json() == []
