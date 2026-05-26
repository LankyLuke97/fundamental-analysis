import { useCallback, useEffect, useRef, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'

const DATA_ENDPOINT = 'http://localhost:8000';
const FILTERS = {ticker: '', dateFrom: '', dateTo: '',}
const App = () => {
    const [transactionItems, setTransactionItems] = useState([]);
    const [filters, setFilters] = useState(FILTERS)

    const fetchEndpoint = useCallback(async () => {
        try {
            const response = await fetch(`${DATA_ENDPOINT}/transactions`);
            const result = await response.json();
            setTransactionItems(result);
        } catch (error) {
            console.log(`Error while downloading from ${DATA_ENDPOINT}:\n`, error);
        }
    }, []);

    useEffect(() => {fetchEndpoint()}, [fetchEndpoint]);

    return (
        <>
            <List items={transactionItems} />
        </>
    );
};

const List = ({items, children}) => (
    <>
    <ul>
        {items.map(item => <ListItem key={item.id} item={item} />)}
    </ul>
    </>
);

const ListItem = ({item, children}) => (
    <>
        <li className="listBroad">
            <span style={{ width: '20%' }}>{item.ticker}</span>&nbsp;
            <span style={{ width: '15%' }}>{item.price}</span>&nbsp;
            <span style={{ width: '15%' }}>{item.quantity}</span>&nbsp;
            <span style={{ width: '50%' }}>{item.timestamp}</span>&nbsp;
        </li>
    </>
);

export default App

