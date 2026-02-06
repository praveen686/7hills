use quantlaxmi_core::{connector::MarketConnector, connector::MockConnector};

#[tokio::test]
async fn test_mock_connector_integration() {
    let _bus = quantlaxmi_core::EventBus::new(100);
    let connector = MockConnector::new("TestMock");

    assert_eq!(connector.name(), "TestMock");

    // Test that it runs and stops without error
    let result = connector.run().await;
    assert!(result.is_ok());

    connector.stop();
}
